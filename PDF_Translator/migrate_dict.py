# -*- coding: utf-8 -*-
"""
garment_dict.json 구조 마이그레이션
기존: {"한글": "번역"}
신규: {"한글": {"full": "번역", "abbr": ""}}
"""
import json
import os

# 기존 ABBREVIATIONS (하드코딩된 약어들)
EXISTING_ABBR = {
    "Garment Matching": "G.M",
    "G Matching": "G.M",
    "Accessory Matching": "A.M",
    "A Matching": "A.M",
    "Consumption": "Cons.",
    "NaturalZipper": "Nat.Zip",
    "Natural Zipper": "Nat.Zip",
    "FrontZipper": "Fr.Zip",
    "Front Zipper": "Fr.Zip",
    "SidePocket": "Side Pkt",
    "Side Pocket": "Side Pkt",
    "Factory Handling": "Fact.Hdl",
    "Hood/Hem": "Hd/Hm",
    # 추가 약어 (긴 단어들)
    "Chest Circumference": "Chest Circ.",
    "Waist Circumference": "Waist Circ.",
    "Hip Circumference": "Hip Circ.",
    "Neck Circumference": "Neck Circ.",
    "Hem Circumference": "Hem Circ.",
    "Color Combination": "Color Comb.",
    "Fusible Interlining": "Fus. Interl.",
    "Detachable Hood": "Det. Hood",
    "Detachable Type": "Det. Type",
    "Flat-felled Seam": "Flat-fell",
    "Seam Allowance": "Seam Allow.",
    "Quality Check": "QC",
    "Needle Detection": "Needle Det.",
    "Sewing Machine": "Sew. Mach.",
    "Sewing Spec": "Sew. Spec",
    "String Stopper": "Str. Stop.",
    "Snap Button": "Snap Btn",
    "Elastic Band": "Elast. Band",
    "Shoulder Pad": "Shldr Pad",
    "Sleeve Length": "Slv. Len.",
    "Sleeve Width": "Slv. Width",
    "Total Length": "Total Len.",
    "Back Length": "Back Len.",
    "Armhole Depth": "Armhole Dep.",
    "Water Repellent": "Water Rep.",
    "UV Protection": "UV Prot.",
    "Anti-static": "Anti-stat.",
    "Moisture Absorption": "Moist. Abs.",
    "Heat Generation": "Heat Gen.",
    "Style Number": "Style No.",
}

def migrate():
    dict_file = os.path.join(os.path.dirname(__file__), "garment_dict.json")
    
    # 백업
    backup_file = dict_file.replace(".json", "_backup.json")
    
    with open(dict_file, 'r', encoding='utf-8') as f:
        old_data = json.load(f)
    
    # 백업 저장
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(old_data, f, ensure_ascii=False, indent=2)
    print(f"백업 완료: {backup_file}")
    
    # 마이그레이션
    new_data = {}
    for lang, terms in old_data.items():
        new_data[lang] = {}
        for korean, translation in terms.items():
            # 기존 약어가 있으면 사용, 없으면 빈 문자열
            abbr = EXISTING_ABBR.get(translation, "")
            new_data[lang][korean] = {
                "full": translation,
                "abbr": abbr
            }
    
    # 저장
    with open(dict_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    # 통계
    total_terms = sum(len(terms) for terms in new_data.values())
    terms_with_abbr = sum(
        1 for lang_terms in new_data.values() 
        for term in lang_terms.values() 
        if term.get("abbr")
    )
    
    print(f"마이그레이션 완료!")
    print(f"총 용어: {total_terms}개")
    print(f"약어 있는 용어: {terms_with_abbr}개")

if __name__ == "__main__":
    migrate()
