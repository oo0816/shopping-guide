"""Conservative type clarification and evidence-based suitability ordering."""

import copy
import re

from app.retrieval import normalize, tokenize

# Category-level language aliases, never product IDs or per-question answers.
TYPE_ALIASES = (
    (("跑步鞋", "徒步鞋", "篮球鞋", "男鞋", "女鞋"), ("鞋", "跑鞋")),
    (("笔记本电脑", "平板电脑"), ("电脑", "笔记本", "平板", "laptop", "tablet")),
    (("智能手机",), ("手机", "smartphone")),
    (("真无线耳机", "耳机音箱"), ("耳机", "音箱", "headphones")),
    (("短袖T恤", "速干T恤", "卫衣", "男装", "女装"), ("衣服", "t恤", "卫衣", "衬衫")),
    (("精华", "面霜", "洁面", "化妆水", "眼霜", "面膜", "防晒"), ("护肤品", "护肤", "精华", "面霜")),
    (("背包", "箱包"), ("背包", "书包", "行李箱", "包包")),
    (("男士腕表", "女士腕表", "智能穿戴"), ("手表", "腕表")),
    (("厨房配件", "厨房餐具"), ("餐具", "厨具", "水杯", "饭盒", "炒锅")),
)
OPTIONAL_QUESTION = re.compile(r"性别|男鞋|女鞋|男款|女款|男士|女士|男性|女性|男生|女生|尺码|鞋码|颜色|预算|价位|品牌")
GENERIC_TERMS = {"希望", "一些", "适合", "用于", "最好", "一点", "需要", "比较", "要求"}
NEGATIVE = re.compile(r"不建议|不适合|不推荐|不适用|不宜|不支持|不能|无法|避免|不要")
DOUBLE_NEGATIVE = re.compile(r"(?:并非|不是|没有|不会|并不)(?:完全)?(?:不适合|不建议|不推荐|不能)")
POSITIVE = re.compile(r"适合|适用|合适|可用于|用于|能穿|轻便|轻量|透气|舒适")


def explicit_product_type(query, needs, context):
    value = normalize(query)
    selected = set(needs["sub_categories"])
    if any(normalize(category) in value for category in selected):
        return True
    for categories, aliases in TYPE_ALIASES:
        if selected.intersection(categories) and any(normalize(alias) in value for alias in aliases):
            return True
    keyword = normalize(needs["keywords"].strip())
    brands = {normalize(brand) for brand in context["brands"]}
    # A concrete series/name is usable, but a brand or restated soft preference is not a type.
    if keyword and keyword in value and keyword not in brands and not any(
        keyword in normalize(preference) for preference in needs["preferences"]
    ):
        return True
    return False


def settle_optional_questions(needs):
    result = copy.deepcopy(needs)
    if (result["sub_categories"] or result["keywords"].strip()) and not result["unsupported_constraints"]:
        result["clarification_questions"] = [
            question for question in result["clarification_questions"]
            if not OPTIONAL_QUESTION.search(question)
        ]
    return result


def initial_clarification_policy(needs, query, context):
    result = copy.deepcopy(needs)
    if not explicit_product_type(query, result, context):
        result["sub_categories"] = []
        result["keywords"] = ""
        result["clarification_questions"] = ["你想选购哪类商品？"]
        return result
    return settle_optional_questions(result)


def assess_suitability(needs, passages):
    """Explicit local wording only: limitations are not universal disqualifications."""
    support, caution = [], []
    for preference in needs["preferences"]:
        # Avoid interpreting a negated preference as a positive keyword match.
        if re.search(r"不要|不想|不喜欢|避免", preference):
            continue
        terms = set(tokenize(preference)) - GENERIC_TERMS
        if not terms:
            continue
        for hit in passages:
            if hit["section"] == "商品评价":
                continue  # One user's review does not determine general suitability.
            body = hit["text"].partition("\n")[2] or hit["text"]
            lines = body.splitlines()
            if lines and lines[0].startswith("商品："):
                body = "\n".join(lines[1:])
            for clause in re.split(r"[。！？；;\n，,]", body):
                clause = clause.strip()
                value = normalize(clause)
                if not clause or not terms.intersection(tokenize(value)) or DOUBLE_NEGATIVE.search(value):
                    continue
                signal = {"preference": preference, "chunk_id": hit["chunk_id"], "quote": clause[:400]}
                if NEGATIVE.search(value):
                    caution.append({**signal, "effect": "caution"})
                elif POSITIVE.search(value):
                    support.append({**signal, "effect": "support"})
    signals = []
    for signal in caution + support:
        if not any(old["chunk_id"] == signal["chunk_id"] and old["quote"] == signal["quote"] for old in signals):
            signals.append(signal)
    status = "caution" if caution else ("supported" if support else "unknown")
    return {
        "status": status,
        "label": {"supported": "用途有资料支持", "unknown": "用途仍需确认", "caution": "资料提示用途限制"}[status],
        "rank": {"supported": 2, "unknown": 1, "caution": 0}[status],
        "signals": signals[:2],
        "method": "explicit_wording_rules_v1",
        "semantic_entailment_verified": False,
    }
