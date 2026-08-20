from __future__ import annotations

from difflib import SequenceMatcher

# All 76 provinces plus the Bangkok special administrative area, in Thai script.
THAI_PROVINCES: tuple[str, ...] = (
    "กรุงเทพมหานคร",
    "อำนาจเจริญ",
    "อ่างทอง",
    "บึงกาฬ",
    "บุรีรัมย์",
    "ฉะเชิงเทรา",
    "ชัยนาท",
    "ชัยภูมิ",
    "จันทบุรี",
    "เชียงใหม่",
    "เชียงราย",
    "ชลบุรี",
    "ชุมพร",
    "กาฬสินธุ์",
    "กำแพงเพชร",
    "กาญจนบุรี",
    "ขอนแก่น",
    "กระบี่",
    "ลำปาง",
    "ลำพูน",
    "เลย",
    "ลพบุรี",
    "แม่ฮ่องสอน",
    "มหาสารคาม",
    "มุกดาหาร",
    "นครนายก",
    "นครปฐม",
    "นครพนม",
    "นครราชสีมา",
    "นครสวรรค์",
    "นครศรีธรรมราช",
    "น่าน",
    "นราธิวาส",
    "หนองบัวลำภู",
    "หนองคาย",
    "นนทบุรี",
    "ปทุมธานี",
    "ปัตตานี",
    "พังงา",
    "พัทลุง",
    "พะเยา",
    "เพชรบูรณ์",
    "เพชรบุรี",
    "พิจิตร",
    "พิษณุโลก",
    "พระนครศรีอยุธยา",
    "แพร่",
    "ภูเก็ต",
    "ปราจีนบุรี",
    "ประจวบคีรีขันธ์",
    "ระนอง",
    "ราชบุรี",
    "ระยอง",
    "ร้อยเอ็ด",
    "สระแก้ว",
    "สกลนคร",
    "สมุทรปราการ",
    "สมุทรสาคร",
    "สมุทรสงคราม",
    "สระบุรี",
    "สตูล",
    "สิงห์บุรี",
    "ศรีสะเกษ",
    "สงขลา",
    "สุโขทัย",
    "สุพรรณบุรี",
    "สุราษฎร์ธานี",
    "สุรินทร์",
    "ตาก",
    "ตรัง",
    "ตราด",
    "อุบลราชธานี",
    "อุดรธานี",
    "อุทัยธานี",
    "อุตรดิตถ์",
    "ยะลา",
    "ยโสธร",
)


def closest_province(text: str, threshold: float = 0.6) -> str | None:
    """Fuzzy-match OCR'd text against the real list of Thai provinces, returning
    the canonical (correctly-spelled) name if it's a plausible match, else None.
    Used to tell an actual (if OCR-garbled) province name apart from unrelated
    text -- e.g. dealer/district branding printed on a plate frame."""
    if not text:
        return None
    best_name, best_ratio = None, 0.0
    for name in THAI_PROVINCES:
        ratio = SequenceMatcher(None, text, name).ratio()
        if ratio > best_ratio:
            best_name, best_ratio = name, ratio
    return best_name if best_ratio >= threshold else None
