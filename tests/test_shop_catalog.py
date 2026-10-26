from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import unittest

from app.catalog import SCHEMA
from app.shop_catalog import ShopCatalogProvider


class LiveCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'catalog.sqlite'
        with closing(sqlite3.connect(self.path)) as db:
            db.executescript(SCHEMA)
            db.execute("INSERT INTO products VALUES ('p1','鞋','A','服饰','跑步鞋',90000,'x','hash','鞋')")
            db.execute("INSERT INTO skus VALUES ('s1','p1','{}',90000)")
            db.commit()
        self.sku = {'skuId': 's1', 'productId': 'p1', 'price': '99.00',
                    'stockQuantity': 2, 'stockStatus': 'in_stock', 'specs': {}}
        self.response = {'items': [{'product_id': 'p1', 'title': '鞋', 'brand': 'A', 'skus': [self.sku]}],
                         'unavailable_product_ids': [], 'observed_at': '2026-10-26T00:00:00Z'}
        self.provider = ShopCatalogProvider('http://unused', self.path, fetch=lambda ids: self.response)
        self.needs = {'sub_categories': ['跑步鞋'], 'keywords': '', 'brand': None,
                      'excluded_brands': [], 'min_price': None, 'max_price': 100}

    def test_price_drop_enters_budget_despite_old_snapshot_price(self):
        items, truncated = self.provider(self.needs)
        self.assertEqual(items[0]['matched_skus'][0]['price'], 99)
        self.assertFalse(truncated)

    def test_price_rise_excludes_candidate(self):
        self.sku['price'] = '100.01'
        self.assertEqual(self.provider(self.needs)[0], [])

    def test_zero_stock_excludes_candidate(self):
        self.sku['stockQuantity'] = 0
        self.assertEqual(self.provider(self.needs)[0], [])

    def test_unavailable_product_excluded(self):
        self.response['items'] = []
        self.response['unavailable_product_ids'] = ['p1']
        self.assertEqual(self.provider(self.needs)[0], [])

    def test_wrong_sku_owner_blocked(self):
        self.sku['productId'] = 'p2'
        with self.assertRaises(ValueError):
            self.provider(self.needs)

    def test_changed_specs_blocked(self):
        self.sku['specs'] = {'尺码': '42'}
        with self.assertRaises(ValueError):
            self.provider(self.needs)

    def test_partial_response_is_not_empty_search(self):
        self.response['items'] = []
        with self.assertRaises(ValueError):
            self.provider(self.needs)

    def test_failure_does_not_fall_back_to_snapshot(self):
        def fail(ids):
            raise ValueError('offline')
        self.provider.fetch = fail
        with self.assertRaises(ValueError):
            self.provider(self.needs)


if __name__ == '__main__':
    unittest.main()
