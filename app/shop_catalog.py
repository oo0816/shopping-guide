"""Live SKU filtering over Go catalog data; local documents remain the evidence base."""
from contextlib import closing
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.catalog import connect_catalog, money_cents
from app.retrieval import normalize


class ShopCatalogProvider:
    def __init__(self, base_url, db_path, fetch=None):
        self.base_url = base_url.rstrip('/')
        self.db_path = db_path
        self.fetch = fetch or self._fetch

    def _fetch(self, ids):
        url = self.base_url + '/products/sku-snapshots?' + urlencode({'product_ids': ','.join(ids)})
        try:
            with urlopen(Request(url, headers={'Accept': 'application/json'}), timeout=8) as response:
                payload = response.read(2_000_001)
                if len(payload) > 2_000_000:
                    raise ValueError('商品接口响应过大。')
                return json.loads(payload)
        except (OSError, ValueError) as exc:
            raise ValueError('实时商品查询失败，请稍后重试；没有使用旧价格继续推荐。') from exc

    def __call__(self, needs, db_path=None, *, price_ceiling=None,
                 excluded_product_ids=None, focus_product_id=None):
        maximum = needs['max_price']
        if price_ceiling is not None:
            maximum = min(maximum, price_ceiling) if maximum is not None else price_ceiling
        lower = money_cents(needs['min_price']) if needs['min_price'] is not None else None
        upper = money_cents(maximum) if maximum is not None else None
        if lower is not None and upper is not None and lower > upper:
            return [], False
        excluded = set(excluded_product_ids or [])
        # No price filtering or pagination on the stale SQLite snapshot: a SKU
        # formerly outside budget may now have a valid live price.
        with closing(connect_catalog(self.db_path)) as db:
            products = [dict(row) for row in db.execute('SELECT * FROM products ORDER BY product_id')]
            local_skus = {row['sku_id']: dict(row) for row in db.execute('SELECT * FROM skus')}
        products = [p for p in products if
            p['product_id'] not in excluded and
            (not focus_product_id or p['product_id'] == focus_product_id) and
            (not needs['sub_categories'] or p['sub_category'] in needs['sub_categories']) and
            (not needs['brand'] or p['brand'] == needs['brand']) and
            p['brand'] not in needs['excluded_brands'] and
            all(term in normalize(p['search_text']) for term in normalize(needs['keywords']).split())]
        found = []
        for offset in range(0, len(products), 30):
            batch = products[offset:offset + 30]
            requested = {p['product_id'] for p in batch}
            response = self.fetch([p['product_id'] for p in batch])
            try:
                live = {p['product_id']: p for p in response['items']}
                unavailable = set(response['unavailable_product_ids'])
                if (len(live) != len(response['items']) or set(live) & unavailable or
                        set(live) | unavailable != requested):
                    raise ValueError('商品接口返回不完整或包含额外商品。')
                for product in batch:
                    current = live.get(product['product_id'])
                    if current is None:
                        continue
                    if current['title'] != product['title'] or current['brand'] != product['brand']:
                        raise ValueError('商品资料已变化，需要同步证据后再推荐。')
                    matched = []
                    for sku in current['skus']:
                        original = local_skus.get(sku['skuId'])
                        if not original:
                            continue  # New SKU has no verified local evidence yet.
                        if (sku['productId'] != product['product_id'] or
                                original['product_id'] != product['product_id'] or
                                sku['specs'] != json.loads(original['properties_json'])):
                            raise ValueError('SKU归属或规格已变化，需要同步资料后再推荐。')
                        cents = money_cents(sku['price'])
                        if (sku['stockQuantity'] <= 0 or sku['stockStatus'] != 'in_stock' or
                                (lower is not None and cents < lower) or
                                (upper is not None and cents > upper)):
                            continue
                        matched.append({'sku_id': sku['skuId'], 'properties': sku['specs'],
                                        'price': cents / 100, 'stock_quantity': sku['stockQuantity'],
                                        'observed_at': response['observed_at'], 'price_source': 'shop_live'})
                    if matched:
                        matched.sort(key=lambda s: (s['price'], s['sku_id']))
                        found.append({**product, 'base_price': matched[0]['price'],
                                      'matched_skus': matched, 'trust_mode': 'assumed_trusted'})
            except (KeyError, TypeError) as exc:
                raise ValueError('实时商品响应不符合约定，停止本轮推荐。') from exc
        found.sort(key=lambda p: (p['matched_skus'][0]['price'], p['product_id']))
        return found, False
