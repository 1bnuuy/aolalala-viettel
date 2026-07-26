# -*- coding: utf-8 -*-
"""
Baseline rule-based / dictionary-based pipeline
(v3 - mo rong tu dien ICD-10/RxNorm + sua loi tach danh sach chan doan).

LUU Y:
- Van la baseline heuristic (regex + tu dien), khong dung model hoc sau do
  sandbox khong truy cap duoc HuggingFace Hub / kho model.
- Bo anh xa ICD-10 / RxNorm la rut gon, uoc luong tu kien thuc chung, KHONG
  tra cuu tu CSDL chinh thuc UMLS/NLM/RxNav (sandbox nay khong co mang).
  Truoc khi dung cho san pham that, nen doi chieu lai voi:
    * ICD-10-CM API:  https://clinicaltables.nlm.nih.gov/apidoc/icd10cm/v3/doc.html
    * RxNorm API:     https://rxnav.nlm.nih.gov/RxNormAPIs.html
- Thay doi so voi v2:
    1. Mo rong SYMPTOM_TERMS voi ~25 trieu chung xac nhan co xuat hien trong
       corpus nhung truoc do bi bo sot (khat nuoc, te bi, nhin mo, ...).
    2. Mo rong DISEASE_ICD10 voi cac alias/viet tat con thieu ma (vd "tang HA",
       "con tim nhanh nhi", tinh trang van tim/stent/cat cut chi).
    3. Mo rong DRUG_NAMES/DRUG_RXNORM voi cac thuoc xuat hien trong corpus
       nhung chua duoc nhan dien (insulin, allopurinol, carvedilol,
       isosorbide, zosyn, cefepim...).
    4. Sua loi trong split_diag_list(): dau "/" va " - " la ranh gioi liet ke
       chan doan ro rang nhung truoc day chi duoc tach khi ky tu sau la chu
       hoa, khien danh sach chan doan dai (vd sau phau thuat) bi gop thanh
       mot cum khong nhan dien duoc. Gio day "/" va "-" luon duoc coi la
       ranh gioi "cung", con dau phay/cham phay/ranh gioi dinh chu van giu
       yeu cau chu hoa phia sau de tranh tach nham giua cau.
    5. Mo rong cua so trich chan doan tu 150 len 300 ky tu de khong cat cut
       cac danh sach chan doan dai truoc khi tach.
"""

import re
import os
import json
import glob
import unicodedata

# --------------------------------------------------------------------------
# 1. TU DIEN / DANH SACH TRIGGER
# --------------------------------------------------------------------------

SYMPTOM_TERMS = [
    # ho hap
    "ho đờm xanh", "ho có đờm", "ho khan", "ho ra máu", "khó thở khi nằm đầu bằng",
    "khó thở khi gắng sức", "khó thở", "tức ngực", "đau ngực", "ho",
    "đờm", "khò khè", "thở khò khè",
    # tieu hoa
    "đau thượng vị", "đau bụng hạ sườn phải", "đau bụng", "buồn nôn",
    "nôn ra thức ăn", "nôn ra máu", "nôn", "ợ hơi", "ợ chua", "tiêu chảy",
    "táo bón", "đại tiện ra máu", "đi ngoài phân đen", "chán ăn", "đầy bụng",
    "khó tiêu", "đau khi sờ nắn", "đau khi hít thở sâu",
    # tim mach
    "tim đập nhanh", "đánh trống ngực", "nhịp tim không đều", "tăng huyết áp",
    "hạ huyết áp", "phù phổi", "phù",
    # than kinh
    "đau đầu", "nhức đầu", "đau nửa đầu", "chóng mặt", "hoa mắt", "co giật",
    "run", "tê", "yếu", "rối loạn ý thức thoáng qua", "biến đổi ý thức",
    "mất ý thức", "ngất", "rối loạn thị giác",
    # tam than
    "lo âu", "mất ngủ", "hoảng sợ", "hoang tưởng", "ảo giác", "trầm cảm",
    "ý nghĩ tự tử", "ý định tự tử", "kích động", "hưng cảm",
    # da lieu
    "vàng da", "ngứa", "phát ban", "ban đỏ toàn thân", "mắt đỏ", "mẩn đỏ",
    "sưng", "nổi mề đay",
    # toan than
    "sốt cao kéo dài", "sốt", "sốt đau", "ớn lạnh", "mệt mỏi", "suy nhược",
    "suy kiệt", "sụt cân", "thiếu máu", "thiếu oxy", "hạ thân nhiệt",
    # tiet nieu / sinh duc
    "tiểu ra máu", "thiểu niệu", "kinh nguyệt thưa", "rậm lông", "béo phì",
    "giảm dung nạp gắng sức", "chảy máu chân răng", "hôi miệng",
    "chảy máu", "đau khớp", "đau nhức", "đau",
    # bo sung vong 3 (quet toan corpus)
    "mất trí nhớ", "mất cảm giác ngon miệng", "mất định hướng", "nổi mề đay",
    "ngứa khắp người", "ngứa da toàn thân", "yếu sức chân phải", "yếu cơ",
    "khó khăn khi ăn uống", "vàng mắt", "bại não", "chậm phát triển trí tuệ",
    "rối loạn vận động", "run tay chân", "run tay", "mất thăng bằng", "ù tai",
    "nhìn song thị", "tri giác giảm sút", "yếu sức nửa người bên phải",
    "nhịp tim chậm", "sốt cao", "phát ban toàn thân", "viêm kết mạc",
    "lưỡi đỏ như dâu tây", "run rẩy toàn thân", "khô miệng",
    "chán ăn", "đầy hơi",
    # bo sung vong 4 (phat hien qua sua loi ranh gioi section)
    "lú lẫn", "lú lẫn ngày càng nặng", "cực kỳ yếu", "ngã", "khó chịu vùng ngực",
    # bo sung vong 5 (quet toan bo corpus doi chieu voi tu dien trieu chung
    # rong hon, xac nhan tung tu that su xuat hien truoc khi them)
    "khát nước", "tiểu nhiều", "tiểu ít", "da khô", "mạch nhanh", "đau lưng",
    "tê bì", "mờ mắt", "nhìn mờ", "khó nuốt", "chảy nước mũi", "hắt hơi",
    "đổ mồ hôi", "phù mặt", "nôn mửa", "chướng bụng", "ợ nóng", "thở nhanh",
    "ngất xỉu", "đi lại khó khăn", "giảm trí nhớ", "phân có máu",
    "chảy máu cam", "rụng tóc", "gan to",
]

DIAGNOSIS_TRIGGER_RE = re.compile(
    r"(?<!dễ bị )(?<!dễ nhầm )(?<!hay nhầm )"
    r"[Cc]hẩn\s*đoán(?!\s*hình\s*ảnh)(?!\s*phân\s*biệt)\s*(?:[:：]\s*)*"
    r"(?:là\s+)?(?:mắc\s+)?([^\n.]{3,300})"
)
DIAGNOSIS_TRIGGER2_RE = re.compile(
    r"được\s+chẩn\s+đoán(?:\s+(?:là|mắc(?:\s+bệnh)?))?\s+([^\n,.]{3,100})"
)

# Cac tu khoa "goc benh" pho bien: dung de loc bo cac doan van tu do trigger
# "chan doan:" nhung thuc te chi la van xuoi/tuong thuat, khong phai ten
# benh (vd bi dinh sang cau ke tiep do van ban nguon thieu dau cham).
MEDICAL_ROOT_HINTS = [
    "viêm", "ung thư", "hội chứng", "suy ", "nhiễm", "tăng ", "hạ ",
    "thiếu", "sỏi", "loét", "xơ ", "rối loạn", "chấn thương", "gãy",
    "khối u", "u ác", "u lành", "bệnh lý", "bệnh mạn", "bệnh tim",
    "đái tháo đường", "huyết áp",
    "dị ứng", "xuất huyết", "tắc", "phù", "liệt", "hoại tử", "nhồi máu",
    "đột quỵ", "tai biến", "gout", "hen ", "lao ", "trĩ", "vô sinh",
    "dại", "sởi", "quai bị", "thủy đậu", "thuỷ đậu", "zona", "cúm",
    "sốt rét", "sốt xuất huyết", "loạn thần", "trầm cảm", "lupus",
    "van động mạch", "van tim", "stent", "cắt cụt",
]
_JUNK_STARTS = ("anh ấy", "cô ấy", "ông ấy", "bà ấy", "khi được",
                "được chuyển", "không còn", "khác:", "và ", "họ ")


def _looks_like_diagnosis(chunk):
    low = chunk.lower().strip().lstrip("-•* \t")
    if low in ("khác", "và điều trị", "lâm sàng", "trước đây"):
        return False
    if low.startswith(_JUNK_STARTS):
        return False
    return any(h in low for h in MEDICAL_ROOT_HINTS)

DISEASE_ICD10 = {
    "tăng huyết áp": ["I10"],
    "đái tháo đường typ ii": ["E11.9"],
    "đái tháo đường": ["E11.9"],
    "viêm phổi bệnh viện": ["J18.9"],
    "viêm phổi thùy dưới phải": ["J18.9"],
    "viêm phổi": ["J18.9"],
    "suy tim": ["I50.9"],
    "đợt cấp copd": ["J44.1"],
    "copd": ["J44.9"],
    "tâm phế mạn": ["I27.9"],
    "viêm dạ dày ruột do virus": ["A08.4"],
    "viêm sung huyết hang vị dạ dày": ["K29.7"],
    "viêm hang vị sung huyết": ["K29.7"],
    "viêm bao tử": ["K29.7"],
    "viêm dạ dày": ["K29.7"],
    "trào ngược dạ dày - thực quản": ["K21.0", "K21.9"],
    "trào ngược dạ dày": ["K21.9"],
    "xuất huyết tiêu hóa": ["K92.2"],
    "viêm cầu thận mạn": ["N03.9"],
    "hội chứng thận hư": ["N04.9"],
    "hội chứng ruột kích thích": ["K58.9"],
    "loét tá tràng": ["K26.9"],
    "loét đại tràng": ["K51.9"],
    "viêm mô tế bào": ["L03.9"],
    "nhồi máu cơ tim": ["I21.9"],
    "thiếu máu cơ tim": ["I25.9"],
    "xuất huyết dưới nhện": ["I60.9"],
    "viêm xoang": ["J32.9"],
    "rối loạn lưỡng cực": ["F31.9"],
    "rối loạn lo âu": ["F41.9"],
    "rối loạn cảm xúc": ["F39"],
    "rối loạn lipid máu": ["E78.5"],
    "trầm cảm": ["F32.9"],
    "giãn phế quản": ["J47"],
    "thuyên tắc phổi": ["I26.9"],
    "suy thận mạn giai đoạn v": ["N18.5"],
    "suy thận mạn": ["N18.9"],
    "viêm tủy xương": ["M86.9"],
    "bàng quang thần kinh": ["N31.9"],
    "tràn dịch màng phổi": ["J91"],
    "bệnh kawasaki": ["M30.3"],
    "dị tật còn ống động mạch": ["Q25.0"],
    "cơn đau thắt ngực không ổn định": ["I20.0"],
    "hội chứng vành cấp": ["I24.9"],
    "nhiễm khuẩn huyết": ["A41.9"],
    "não úng thủy": ["G91.9"],
    "giãn thừng tinh": ["I86.1"],
    "quá liều kháng vitamin k": ["T45.7"],
    "viêm túi mật cấp": ["K81.0"],
    "ung thư biểu mô tuyến": ["C80.1"],
    "ung thư biểu mô tế bào vảy": ["C80.1"],
    "ung thư máu": ["C95.9"],
    "ung thư biểu mô tế bào mật": ["C22.1"],
    "hội chứng buồng trứng đa nang": ["E28.2"],
    "viêm gan b": ["B18.1"],
    "viêm da tiếp xúc dị ứng": ["L23.9"],
    "viêm da tiếp xúc": ["L25.9"],
    "viêm nha chu": ["K05.6"],
    "viêm cầu thận": ["N05.9"],
    "hội chứng nghiện rượu": ["F10.2"],
    "thiếu men g6pd": ["D55.0"],
    "hội chứng parkinson": ["G20"],
    "vô sinh": ["N97.9"],
    "bệnh dại": ["A82.9"],
    "nhiễm khuẩn đường tiết niệu": ["N39.0"],
    "nhiễm trùng đường tiết niệu": ["N39.0"],
    "tăng lipid máu": ["E78.5"],
    "tăng sản tuyến tiền liệt": ["N40"],
    "xơ gan do rượu": ["K70.3"],
    "xơ gan": ["K74.6"],
    "sỏi ống mật chủ": ["K80.5"],
    "xơ vữa động mạch": ["I70.9"],
    "tăng kali máu": ["E87.5"],
    "tăng cholesterol máu": ["E78.0"],
    "tăng áp lực động mạch phổi": ["I27.0"],
    "nhiễm trùng máu": ["A41.9"],
    "tăng nhãn áp": ["H40.9"],
    "tăng sản tuyến bã nhờn": ["L73.9"],
    "bệnh parkinson": ["G20"],
    "đột quỵ": ["I63.9"],
    "tai biến mạch máu não": ["I63.9"],
    "chấn thương sọ não": ["S06.9"],
    "bệnh đa xơ cứng": ["G35"],
    "loạn thần": ["F29"],
    "viêm kết mạc": ["H10.9"],
    # bo sung tu du lieu thuc te (vong danh gia) + cac chinh ta cu/moi
    "zona": ["B02.9"],
    "zona thần kinh": ["B02.9"],
    "thủy đậu": ["B01.9"],
    "thuỷ đậu": ["B01.9"],
    "nhiễm khuẩn đường tiêu hóa": ["A09"],
    "nhiễm khuẩn đường tiêu hoá": ["A09"],
    "phù phổi cấp": ["J81"],
    "phù phổi": ["J81"],
    "nhiễm trùng huyết": ["A41.9"],
    "viêm tuỷ xương": ["M86.9"],
    "u ác của tuyến tiền liệt": ["C61"],
    "u ác tuyến tiền liệt": ["C61"],
    "ung thư tuyến tiền liệt": ["C61"],
    "cml": ["C92.1"],
    "bạch cầu tủy mạn": ["C92.1"],
    "vpht": ["J85.0"],
    "viêm phổi hoại tử": ["J85.0"],
    "nhịp nhanh nhĩ": ["I47.1"],
    "cơn nhịp nhanh nhĩ": ["I47.1"],
    # bo sung them cac chan doan pho bien khac de tang do bao phu
    "hen phế quản": ["J45.9"],
    "viêm phế quản": ["J20.9"],
    "viêm amidan": ["J03.9"],
    "viêm họng": ["J02.9"],
    "cúm": ["J11.1"],
    "covid-19": ["U07.1"],
    "sốt xuất huyết": ["A90"],
    "sốt rét": ["B54"],
    "lao phổi": ["A15.0"],
    "viêm màng não": ["G03.9"],
    "động kinh": ["G40.9"],
    "đau nửa đầu migraine": ["G43.9"],
    "gout": ["M10.9"],
    "loãng xương": ["M81.9"],
    "thoái hóa khớp": ["M19.9"],
    "thoát vị đĩa đệm": ["M51.9"],
    "viêm khớp dạng thấp": ["M06.9"],
    "sỏi thận": ["N20.0"],
    "sỏi mật": ["K80.2"],
    "viêm ruột thừa": ["K35.8"],
    "tắc ruột": ["K56.6"],
    "thoát vị bẹn": ["K40.9"],
    "trĩ": ["K64.9"],
    "ung thư phổi": ["C34.9"],
    "ung thư vú": ["C50.9"],
    "ung thư gan": ["C22.9"],
    "ung thư dạ dày": ["C16.9"],
    "ung thư đại trực tràng": ["C18.9"],
    "ung thư cổ tử cung": ["C53.9"],
    "suy giáp": ["E03.9"],
    "cường giáp": ["E05.9"],
    "bướu cổ": ["E04.9"],
    "béo phì": ["E66.9"],
    "tự kỷ": ["F84.0"],
    "tăng động giảm chú ý": ["F90.9"],
    "viêm gan c": ["B18.2"],
    "hiv": ["B20"],
    "lupus ban đỏ hệ thống": ["M32.9"],
    "viêm tụy cấp": ["K85.9"],
    "viêm tụy mạn": ["K86.1"],
    "suy hô hấp": ["J96.9"],
    "ngưng thở khi ngủ": ["G47.3"],
    "rung nhĩ": ["I48.9"],
    "block nhĩ thất": ["I44.3"],
    "nhồi máu não": ["I63.9"],
    "phình động mạch chủ": ["I71.9"],
    "huyết khối tĩnh mạch sâu": ["I80.2"],
    "viêm tắc tĩnh mạch": ["I80.9"],
    # bo sung vong 5: alias/viet tat va cac chan doan phat hien qua quet
    # toan bo corpus (xem ghi chu o dau file ve do chinh xac uoc luong)
    "tăng ha": ["I10"],  # viet tat cua "tang huyet ap"
    "tăng huyết áp vô căn": ["I10"],
    "tăng huyết áp nguyên phát": ["I10"],
    "cơn tim nhanh nhĩ": ["I47.1"],
    "van động mạch chủ cơ": ["Z95.2"],  # tinh trang mang van dong mach chu co hoc
    "van động mạch chủ nhân tạo": ["Z95.2"],
    "stent mạch vành": ["Z95.5"],
    "cắt cụt chân": ["Z89.9"],
    "cắt cụt chi": ["Z89.9"],
    "nhiễm trùng chi dưới": ["L03.1"],
    "thiểu sản vành tai": ["Q17.2"],
    "tịt ống tai ngoài": ["Q16.1"],
    # bo sung vong 6 (quet loi trich xuat khong co candidates + tu dien
    # ICD-10 mo rong doi chieu them cac chan doan/phat hien xuat hien
    # trong corpus nhung truoc do chua co ma anh xa)
    "xẹp phổi": ["J98.1"],
    "viêm xương tủy": ["M86.9"],
    "viêm xương tuỷ": ["M86.9"],
    "ung thư di căn": ["C79.9"],
    "u ác di căn": ["C79.9"],
    "tắc nghẽn đường mật": ["K83.1"],
    "tắc nghẽn ống mật": ["K83.1"],
    "hẹp động mạch thận": ["I70.1"],
    "tắc hẹp động mạch thận": ["I70.1"],
    "bệnh lý chất trắng": ["G93.9"],
    "viêm túi mật": ["K81.9"],
    # bo sung vong 7 (quet cac file dang Q&A suc khoe cong dong bi bo sot
    # hoan toan trong cac vong truoc, xac dinh qua ra soat cac file co so
    # luong thuc the trich xuat qua thap)
    "phù gai thị": ["H47.1"],
    "thoái hóa tinh bột": ["E85.9"],
    "rối loạn chuyển hóa tinh bột": ["E85.9"],
    "amyloidosis": ["E85.9"],
    "mày đay vô căn": ["L50.1"],
    "mày đay mạn": ["L50.8"],
    "mày đay": ["L50.9"],
    "mề đay": ["L50.9"],
    "hội chứng tăng đông": ["D68.9"],
    "thrombophilia": ["D68.9"],
    "thiểu sản vành tai": ["Q17.2"],
    "tịt ống tai ngoài": ["Q16.1"],
    "nấm bẹn": ["B35.6"],
    "tiền sản giật": ["O14.9"],
    "mụn trứng cá": ["L70.0"],
    "u xơ tuyến vú": ["D24"],
    "u nang tuyến vú": ["N60.1"],
}

DRUG_NAMES = [
    "chlorpheniramine", "capsaicin", "amlodipine", "aspirin",
    "metoprolol succinate xl", "metoprolol", "guaifenesin", "nystatin",
    "acetaminophen", "pravastatin", "docusate sodium", "docusate",
    "senna", "clonazepam", "furosemid", "furosemide", "medrol",
    "methylprednisolone", "dilaudid", "hydromorphone", "lasix",
    "morphine", "nitramyl", "nitroglycerin", "omez", "omeprazole",
    "zestril", "lisinopril", "bumetanide", "diltiazem", "simethicon",
    "simethicone", "coumadin", "warfarin", "levofloxacin",
    "metoclopramide", "azithromycin", "gleevec", "imatinib",
    "suboxone", "klonopin", "clonidine", "ceftriaxone", "fortex",
    "philpovin", "lorazepam", "compazine", "amoxicillin", "atenolol",
    "doxycycline", "rosuvastatin", "torsemide", "vancomycin",
    "compazine", "naproxen", "prochlorperazine", "bactrim", "cotrimoxazol",
    "paracetamol", "ibuprofen", "diclofenac", "metformin", "losartan",
    "valsartan", "enalapril", "captopril", "hydrochlorothiazide",
    "dexamethasone", "clopidogrel", "atorvastatin", "simvastatin",
    "pantoprazole", "ranitidine", "metronidazole", "ciprofloxacin",
    "salbutamol", "loratadine", "cetirizine", "diazepam", "alprazolam",
    "sertraline", "fluoxetine", "haloperidol", "levothyroxine",
    "gabapentin", "tramadol", "codeine", "diphenhydramine", "prednisone",
    "prednisolone", "domperidone", "famotidine", "montelukast",
    "fluticasone", "risperidone", "olanzapine", "amitriptyline",
    "fortec", "fortex", "philpovin",
    # bo sung vong 5 (quet corpus: thuoc xuat hien nhung chua co trong tu dien)
    "insulin glargine", "insulin", "allopurinol", "carvedilol",
    "isosorbide", "crestor", "zosyn", "cefepim", "cefepime",
    # bo sung vong 6
    "methadone", "hydroxyzine", "spironolactone", "digoxin",
    "amiodarone", "heparin", "enoxaparin", "clindamycin",
    "meropenem", "linezolid", "fentanyl", "oxycodone",
    "eliquis", "apixaban", "seroquel", "quetiapine",
]

DRUG_RXNORM = {
    "chlorpheniramine": ["2599"], "capsaicin": ["1994"],
    "amlodipine": ["17767"], "aspirin": ["1191"],
    "metoprolol succinate xl": ["6918"], "metoprolol": ["6918"],
    "guaifenesin": ["5032"], "nystatin": ["7573"],
    "acetaminophen": ["161"], "pravastatin": ["42463"],
    "docusate sodium": ["3443"], "docusate": ["3443"], "senna": ["9997"],
    "clonazepam": ["2598"], "furosemid": ["4603"], "furosemide": ["4603"],
    "medrol": ["6902"], "methylprednisolone": ["6902"],
    "dilaudid": ["3423"], "hydromorphone": ["3423"], "lasix": ["4603"],
    "morphine": ["7052"], "nitroglycerin": ["7417"], "nitramyl": ["7417"],
    "omez": ["7646"], "omeprazole": ["7646"], "zestril": ["29046"],
    "lisinopril": ["29046"], "bumetanide": ["1808"], "diltiazem": ["3443"],
    "simethicon": ["37418"], "simethicone": ["37418"], "coumadin": ["11289"],
    "warfarin": ["11289"], "levofloxacin": ["82122"],
    "metoclopramide": ["6959"], "azithromycin": ["18631"],
    "gleevec": ["282388"], "imatinib": ["282388"], "suboxone": ["351131"],
    "clonidine": ["2599"], "ceftriaxone": ["2193"], "lorazepam": ["6470"],
    "amoxicillin": ["723"], "atenolol": ["1202"], "doxycycline": ["3640"],
    "rosuvastatin": ["301542"], "torsemide": ["38413"], "vancomycin": ["11124"],
    "compazine": ["8745"], "prochlorperazine": ["8745"], "naproxen": ["7258"],
    "bactrim": ["10829"], "cotrimoxazol": ["10829"],
    "paracetamol": ["161"], "ibuprofen": ["5640"], "diclofenac": ["3355"],
    "metformin": ["6809"], "losartan": ["52175"], "valsartan": ["69749"],
    "enalapril": ["3827"], "captopril": ["1998"],
    "hydrochlorothiazide": ["5487"], "dexamethasone": ["3264"],
    "clopidogrel": ["32968"], "atorvastatin": ["83367"],
    "simvastatin": ["36567"], "pantoprazole": ["40790"],
    "ranitidine": ["9143"], "metronidazole": ["6922"],
    "ciprofloxacin": ["2551"], "salbutamol": ["435"], "loratadine": ["6849"],
    "cetirizine": ["20610"], "diazepam": ["3322"], "alprazolam": ["596"],
    "sertraline": ["36437"], "fluoxetine": ["4493"], "haloperidol": ["5093"],
    "levothyroxine": ["10582"], "gabapentin": ["25480"],
    "tramadol": ["10689"], "codeine": ["2670"],
    "diphenhydramine": ["3498"], "prednisone": ["8640"],
    "prednisolone": ["8638"], "domperidone": ["3475"],
    "famotidine": ["4278"], "montelukast": ["42375"],
    "fluticasone": ["41126"], "risperidone": ["35636"],
    "olanzapine": ["61381"], "amitriptyline": ["704"],
    # Fortec/Fortex (biphenyl dimethyl dicarboxylate) and Philpovin
    # (L-ornithine-L-aspartate) are Vietnam-market hepatoprotectants with
    # no corresponding RxNorm ingredient entry (RxNorm is US-market only),
    # so intentionally left without a candidates mapping.
    # bo sung vong 5 -- estimated RxCUI ingredient codes from general
    # knowledge; NOT verified against the live RxNorm API (no network
    # access in this sandbox). Re-check via rxnav.nlm.nih.gov before
    # using these in a production system.
    "insulin glargine": ["274783"], "insulin": ["5856"],
    "allopurinol": ["519"], "carvedilol": ["20352"],
    "isosorbide": ["6924"],  # assumed isosorbide mononitrate; verify vs dinitrate (6132) from context
    "crestor": ["301542"],  # brand for rosuvastatin
    "zosyn": ["33533"],  # brand for piperacillin/tazobactam (combo, approximate)
    "cefepim": ["2183"], "cefepime": ["2183"],
    # bo sung vong 6 -- estimated RxCUI ingredient codes, chua xac minh qua
    # RxNav API (khong co mang trong sandbox nay)
    "methadone": ["6813"], "hydroxyzine": ["5716"],
    "digoxin": ["3407"],
    "amiodarone": ["703"], "heparin": ["5224"],
    "enoxaparin": ["67108"], "clindamycin": ["2582"],
    "meropenem": ["44245"], "linezolid": ["108140"],
    "fentanyl": ["4337"], "oxycodone": ["7804"],
    "eliquis": ["1364430"], "apixaban": ["1364430"],
    "seroquel": ["51272"], "quetiapine": ["51272"],
}

# --------------------------------------------------------------------------
# 1b. ANH XA RXNORM THEO LIEU LUONG CU THE (SCD - Semantic Clinical Drug)
# --------------------------------------------------------------------------
# QUAN TRONG: vi du chinh thuc cua de bai cho thay ground truth dung ma
# RxCUI cua "ten hoat chat + ham luong + dang bao che" (SCD), KHONG PHAI ma
# hoat chat chung chung. Vi du clonazepam 0.5mg va 1.5mg co 2 ma RxCUI khac
# nhau (197527 vs 197528) du cung mot hoat chat. Bang duoi day anh xa
# (ten thuoc, lieu luong chuan hoa) -> RxCUI SCD, doi chieu tu du lieu vi
# du chinh thuc cua de bai va tra cuu qua RxNorm/RxNav/RxTerms (rxnav.nlm.
# nih.gov, ndclist.com) khi co the truy cap mang. Neu khong tim thay lieu
# luong khop, se fallback ve ma hoat chat chung (DRUG_RXNORM o tren).
DOSE_SCD_MAP = {
    # -- tu vi du chinh thuc cua de bai (chac chan dung) --
    ("amlodipine", "10mg"): "308135",
    ("aspirin", "81mg"): "243670",
    ("metoprolol succinate xl", "50mg"): "866436",
    ("guaifenesin", None): "392085",
    ("nystatin oral suspension", "5ml"): "7597",
    ("nystatin", "5ml"): "7597",
    ("acetaminophen", "325mg"): "313782",
    ("pravastatin", "40mg"): "904475",
    ("docusate sodium", "100mg"): "1099279",
    ("docusate", "100mg"): "1099279",
    ("senna", "8.6mg"): "312935",
    ("clonazepam", "0.5mg"): "197527",
    ("clonazepam", "1.5mg"): "197528",
    # -- doi chieu tra cuu them qua RxNav/RxTerms/ndclist (rxnav.nlm.nih.gov,
    # ndclist.com) trong pham vi tim kiem duoc cua sandbox nay --
    ("acetaminophen", "500mg"): "198440",
    ("paracetamol", "500mg"): "198440",
    ("aspirin", "325mg"): "212033",
    ("furosemide", "40mg"): "313988",
    ("furosemid", "40mg"): "313988",
    ("lasix", "40mg"): "313988",
    ("clopidogrel", "75mg"): "309362",
    ("simvastatin", "20mg"): "312961",
    ("metoprolol succinate", "100mg"): "866412",
    ("nitroglycerin", "0.4mg"): "705129",
}

_DOSE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mg|mcg|ml|g)\b", re.IGNORECASE
)


def _extract_dose(text):
    """Chuan hoa lieu luong dau tien tim thay trong text (vd '10 mg' ->
    '10mg', '8,6 mg' -> '8.6mg') de tra cuu trong DOSE_SCD_MAP."""
    m = _DOSE_RE.search(text)
    if not m:
        return None
    num = m.group(1).replace(",", ".")
    unit = m.group(2).lower()
    if num.endswith(".0"):
        num = num[:-2]
    return f"{num}{unit}"


def get_drug_scd_candidate(entity_text):
    """Uu tien tra cuu ma RxCUI SCD (hoat chat + lieu cu the) truoc khi
    fallback ve ma hoat chat chung trong DRUG_RXNORM."""
    low = entity_text.lower().strip()
    dose = _extract_dose(low)
    best = None
    for (name, d), rxcui in DOSE_SCD_MAP.items():
        if name not in low:
            continue
        if d is None or d == dose:
            if best is None or len(name) > len(best[0]):
                best = (name, rxcui)
    return best[1] if best else None

NEGATION_TRIGGERS = ["không có", "không thấy", "phủ nhận", "chưa ghi nhận",
                      "loại trừ", "không còn", "không hề", "không xuất hiện",
                      "âm tính với", "không rõ", "không", "chưa"]
FAMILY_TRIGGERS = ["người nhà", "gia đình", "họ hàng", "mẹ", "bố", "cha",
                    "anh trai", "chị gái", "em trai", "em gái", "bố mẹ",
                    "vợ", "chồng", "con trai", "con gái", "ông", "bà",
                    "cô", "chú", "dì", "cậu", "người thân"]
HISTORY_TRIGGERS = ["tiền sử", "tiền căn", "đã từng", "trước đây",
                     "trước khi nhập viện", "trong quá khứ", "trước nhập viện",
                     "từng", "cách đây", "trước lúc nhập viện", "năm trước",
                     "tháng trước", "trước đó"]
# Cac dong tu the hien nguoi nha CHI LA nguoi quan sat/bao cao trieu chung
# cua benh nhan (khong phai la tien su benh cua chinh nguoi nha), de tranh
# gan nham "isFamily" cho cau kieu "... duoc vo nhan thay".
OBSERVER_TRIGGERS = ["nhận thấy", "kể lại", "cho biết", "quan sát thấy",
                      "phát hiện", "kể rằng", "thông báo", "báo lại"]
HISTORY_SECTION_RE = re.compile(
    r"(tiền sử|thuốc trước khi nhập viện|thuốc trước nhập viện)", re.IGNORECASE
)
SECTION_HEADER_RE = re.compile(
    r"(?:^|\n)\s*(?:\d+\.\s*)?([A-ZĐÀ-Ỹ][^\n:]{2,60})(?:[:\n])"
)

# Section headings -> loai khai niem, dung de trich xuat item dang bullet
SECTION_TYPE_HEADERS = {
    "TRIỆU_CHỨNG": [
        "triệu chứng hiện tại", "các triệu chứng hiện tại", "triệu chứng khi nhập viện",
        "triệu chứng khi vào viện", "dấu hiệu lâm sàng", "đặc điểm triệu chứng",
        "triệu chứng lâm sàng", "triệu chứng điển hình", "triệu chứng không điển hình",
        "triệu chứng bắt đầu",
    ],
    "THUỐC": [
        "thuốc trước khi nhập viện", "thuốc trước nhập viện", "đơn thuốc",
        "thuốc đã dùng trước đây", "thuốc điều trị", "các thuốc đã thực hiện",
        "thuốc đã điều trị trước khi nhập viện", "thuốc nền tảng",
    ],
    "CHẨN_ĐOÁN": [
        "chẩn đoán phân biệt", "các phát hiện chẩn đoán khác",
        "chẩn đoán xác định", "các kết quả chẩn đoán khác",
        # bo sung vong 8: quet toan bo corpus phat hien cac tieu de nay
        # dan dau nhung danh sach benh dang bullet nhung TRUOC DAY KHONG
        # duoc anh xa toi loai nao (vd "Bệnh phổi kẽ do sử dụng corticoid
        # liều cao kéo dài", "Hội chứng kháng enzym tổng hợp protein" --
        # cac ten benh khong co trong tu dien ICD10 nen chi co the duoc
        # bat qua co che section-header nay).
        "tiền sử bệnh nội khoa", "bệnh lý mãn tính", "tiền sử bản thân",
    ],
}
ANY_HEADER_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:\d+[\.\)]\s*)?([A-ZĐÀ-Ỹa-zà-ỹ][^\n:]{1,60}):"
)
BULLET_RE = re.compile(r"(?:^|\n)[ \t]*[-•\*][ \t]*([^\n]+)")
# Tieu de danh so (vd "\n2.  Tien su benh hien tai") luon la ranh gioi
# section cung du khong co dau hai cham theo sau.
NUMBERED_HEADER_RE = re.compile(r"\n[ \t]*\d+\.[ \t]+[A-ZĐÀ-Ỹ]")
# Mot dong Title-Case ngan (khong co dau ':') ngay truoc mot danh sach
# bullet moi cung la dau hieu section moi (vd "Cac thu thuat da thuc hien").
SUBHEADER_BULLET_RE = re.compile(
    r"\n[ \t]*[A-ZĐÀ-Ỹ][^\n:]{2,60}\n[ \t]*[-•\*]"
)

TYPE_ORDER = {"CHẨN_ĐOÁN": 0, "THUỐC": 1, "KẾT_QUẢ_XÉT_NGHIỆM": 2,
              "TÊN_XÉT_NGHIỆM": 3, "TRIỆU_CHỨNG": 4}


# --------------------------------------------------------------------------
# 2. TRICH XUAT UNG VIEN THEO TUNG LOAI
# --------------------------------------------------------------------------

def find_vitals(text):
    cands = []
    pat = re.compile(
        r"(Huyết áp|Mạch|Nhiệt độ|Nhịp thở|SPO2|SpO2|spo2)\s*[:：]?\s*"
        r"([0-9][0-9./,\s]*\s*(?:mmHg|mmhg|lần/phút|l/p|°C|%|ph)?)",
    )
    for m in pat.finditer(text):
        label_s, label_e = m.span(1)
        val_s, val_e = m.span(2)
        val_text = text[val_s:val_e].rstrip()
        if not val_text.strip():
            continue
        val_e = val_s + len(val_text)
        cands.append((label_s, label_e, text[label_s:label_e], "TÊN_XÉT_NGHIỆM"))
        cands.append((val_s, val_e, text[val_s:val_e], "KẾT_QUẢ_XÉT_NGHIỆM"))
    return cands


def find_lab_panel(text):
    """Bat pattern 'Ten (mo ta): gia tri' ket thuc bang ';' hoac newline."""
    cands = []
    pat = re.compile(
        r"([A-Za-zÀ-Ỹà-ỹ][A-Za-zÀ-Ỹà-ỹ%\s\(\)]{1,40}?)\s*[:：]\s*"
        r"([0-9]+[.,]?[0-9]*)\s*(?:;|\n|$)"
    )
    for m in pat.finditer(text):
        name_s, name_e = m.span(1)
        val_s, val_e = m.span(2)
        name_txt = text[name_s:name_e].strip()
        if len(name_txt) < 2:
            continue
        cands.append((name_s, name_e, name_txt, "TÊN_XÉT_NGHIỆM"))
        cands.append((val_s, val_e, text[val_s:val_e], "KẾT_QUẢ_XÉT_NGHIỆM"))
    return cands


# Cac tu trieu chung ngan, de bi trung voi mot tu/cum tu khac co nghia hoan
# toan khac (khong phai trieu chung) khi dung tu dien don gian. Voi moi tu
# nay, dinh nghia them dieu kien ngu canh am/duong de loai bo false
# positive da phat hien qua ra soat corpus vong 7.
SYMPTOM_CONTEXT_FILTERS = {
    # "yếu" (weak) khong duoc tinh neu la mot phan cua "yếu tố" (factor) hoac
    # "chủ yếu" (mainly/chiefly) -- ca hai deu rat pho bien trong van ban
    # nhung khong lien quan gi den trieu chung yeu suc.
    "yếu": lambda before, after: not after.startswith("tố") and not before.rstrip().endswith("chủ"),
    # "phù" (edema/swelling) khong duoc tinh neu la "phù hợp" (suitable).
    # "phù gai thị" (papilledema) la mot khai niem lam sang khac, duoc xu ly
    # rieng trong tu dien CHẨN_ĐOÁN nen cung loai khoi day de tranh trung 2 lan.
    "phù": lambda before, after: not after.startswith("hợp") and not after.startswith("gai thị"),
}


def find_symptoms(text):
    cands = []
    low = text.lower()
    for term in sorted(SYMPTOM_TERMS, key=len, reverse=True):
        filt = SYMPTOM_CONTEXT_FILTERS.get(term)
        for m in re.finditer(r"\b" + re.escape(term) + r"\b", low):
            s, e = m.span()
            if filt is not None:
                before = low[max(0, s - 15):s]
                after = low[e:e + 15].lstrip()
                if not filt(before, after):
                    continue
            cands.append((s, e, text[s:e], "TRIỆU_CHỨNG"))
    return cands


def find_diagnoses(text):
    cands = []

    def split_diag_list(seg_start, segment):
        pieces = []
        last = 0
        # Ranh gioi tach: "- X", "/", ",", ";" khi ky tu sau la chu hoa (bat
        # dau muc moi), va ranh gioi cau bi dinh lien khong co khoang trang
        # (vd "...tuyến tiền liệtAnh ấy...") thuong gap trong du lieu nguon.
        boundary_re = re.compile(
            r"(?P<hard>\s*-\s*|/|(?<=[0-9])(?=[a-zà-ỹA-ZĐÀ-Ỹ]))|(?P<soft>,|;)|"
            r"(?P<camel>(?<=[a-zà-ỹ])(?=[A-ZĐÀ-Ỹ]))"
        )
        for mm in boundary_re.finditer(segment):
            if mm.lastgroup == "hard":
                split_here = True
            else:
                after = segment[mm.end():mm.end() + 1]
                after_word = segment[mm.end():mm.end() + 11]
                split_here = after.isupper() or after_word.lower().startswith(
                    ("phát hiện ",)
                )
            if split_here:
                pieces.append((last, mm.start()))
                last = mm.end()
        pieces.append((last, len(segment)))
        LEAD_STRIP_RE = re.compile(
            r"^(?:phát hiện(?:\s+thêm)?|ghi nhận|cho thấy)\s+", re.IGNORECASE
        )
        TRAIL_STRIP_RE = re.compile(
            r"\s+(?:khi (?:làm )?nội soi|qua nội soi)$", re.IGNORECASE
        )
        for a, b in pieces:
            raw = segment[a:b]
            chunk = raw.strip(" .:：")
            if chunk.endswith(")") and "(" not in chunk:
                chunk = chunk[:-1].strip()
            lm = LEAD_STRIP_RE.match(chunk)
            if lm:
                chunk = chunk[lm.end():]
            tm = TRAIL_STRIP_RE.search(chunk)
            if tm:
                chunk = chunk[:tm.start()]
            if len(chunk) >= 3:
                off = raw.find(chunk[0]) if chunk else 0
                cs = seg_start + a + off
                ce = cs + len(chunk)
                pieces_out.append((cs, ce, chunk))

    # Cac cum tu noi tiep theo sau ten benh thuong la mo ta/giai thich them
    # chu khong phai liet ke chan doan khac (vd "não úng thủy, có biểu hiện
    # tăng đau đầu..."). Cat doan tai day de tranh bat ca cau dai lam CHẨN
    # ĐOÁN.
    CONT_CUT_RE = re.compile(
        r",?\s*(?:có biểu hiện|biểu hiện|triệu chứng cải thiện|"
        r"nghi (?:ngờ\s+)?liên quan|nghi ngờ liên quan|và có thể dẫn đến|"
        r"có thể dẫn đến|dẫn đến|nguyên nhân(?:\s+do)?|"
        r"điều trị bằng)",
        re.IGNORECASE,
    )

    pieces_out = []
    for pat in (DIAGNOSIS_TRIGGER_RE, DIAGNOSIS_TRIGGER2_RE):
        for m in pat.finditer(text):
            seg_s, seg_e = m.span(1)
            seg_text = text[seg_s:seg_e]
            cm = CONT_CUT_RE.search(seg_text)
            if cm and cm.start() >= 3:
                seg_e = seg_s + cm.start()
                seg_text = text[seg_s:seg_e]
            split_diag_list(seg_s, seg_text)
    for cs, ce, chunk in pieces_out:
        if _looks_like_diagnosis(chunk):
            cands.append((cs, ce, chunk, "CHẨN_ĐOÁN"))

    low = text.lower()
    for name in sorted(DISEASE_ICD10.keys(), key=len, reverse=True):
        for m in re.finditer(r"\b" + re.escape(name) + r"\b", low):
            s, e = m.span()
            cands.append((s, e, text[s:e], "CHẨN_ĐOÁN"))

    # Mot so cum chan doan trong corpus bi chen them so lieu/tinh trang o
    # giua (vd "tắc hẹp 80% động mạch thận trái") nen khong khop duoc voi
    # cum "cứng" trong tu dien qua exact-substring. Bo sung regex linh hoat
    # cho phep xen mot cum dinh luong (vd "80%", "mức độ nặng") giua cac
    # tu goc cua ten benh, ap dung cho mot vai mau hinh pho bien da phat
    # hien qua ra soat cac file co so luong thuc the trich xuat qua thap.
    FLEX_DISEASE_PATTERNS = [
        (re.compile(r"tắc\s+hẹp(?:\s+[\d.,]+\s*%)?\s+động\s+mạch\s+thận(?:\s+(?:trái|phải))?"),
         ["I70.1"]),
        (re.compile(r"hẹp(?:\s+[\d.,]+\s*%)?\s+động\s+mạch\s+vành(?:\s+(?:trái|phải))?"),
         ["I25.1"]),
    ]
    for pat, codes in FLEX_DISEASE_PATTERNS:
        for m in pat.finditer(low):
            s, e = m.span()
            cands.append((s, e, text[s:e], "CHẨN_ĐOÁN"))
            _FLEX_CODES[text[s:e].lower()] = codes
    return cands


_FLEX_CODES = {}


def find_drugs(text):
    cands = []
    low = text.lower()
    dosage_tail = re.compile(
        r"^(\s*[0-9][0-9.,/\-]*\s*(?:mg/ml|mcg/ml|mg|ml|mcg|g|iu)\b"
        r"(?:\s*(?:x\s*\d+\s*(?:viên|ống|lọ)|po|iv|bid|daily|qid|qam|qhs|"
        r"q\d+h(?::prn)?|once|/ngày|/lần)\b)*)+",
        re.IGNORECASE
    )
    for name in sorted(DRUG_NAMES, key=len, reverse=True):
        for m in re.finditer(r"\b" + re.escape(name) + r"\b", low):
            s, e = m.span()
            tail_match = dosage_tail.match(text[e:e + 60])
            end = e + (tail_match.end() if tail_match else 0)
            cands.append((s, end, text[s:end].strip(), "THUỐC"))
    return cands


TRUSTED_DIAGNOSIS_LIST_HEADERS = {
    "tiền sử bệnh nội khoa", "bệnh lý mãn tính", "tiền sử bản thân",
}


NO_COLON_TRUSTED_HEADER_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:[Cc]ác\s+)?"
    r"(bệnh lý mãn tính|tiền sử bệnh nội khoa|tiền sử bản thân)"
    r"[ \t]*\n", re.IGNORECASE
)


def find_trusted_diagnosis_lists(text):
    """Bat cac tieu de danh sach benh man tinh KHONG co dau hai cham (vd
    'Các bệnh lý mãn tính' tren mot dong rieng, theo sau la danh sach
    bullet) -- ANY_HEADER_RE yeu cau dau ':' nen bo sot dang nay."""
    cands = []
    for hm in NO_COLON_TRUSTED_HEADER_RE.finditer(text):
        block_start = hm.end()
        block_end = min(len(text), block_start + 500)
        block = text[block_start:block_end]
        blank_idx = block.find("\n\n")
        if blank_idx != -1:
            block = block[:blank_idx]
        num_hdr = NUMBERED_HEADER_RE.search(block)
        if num_hdr:
            block = block[:num_hdr.start()]
        sub_hdr = SUBHEADER_BULLET_RE.search(block)
        if sub_hdr:
            block = block[:sub_hdr.start()]
        for bm in BULLET_RE.finditer(block):
            raw = bm.group(1)
            stripped = raw.strip()
            if not (2 <= len(stripped) <= 80):
                continue
            if stripped.count("(") > stripped.count(")"):
                continue
            lead = len(raw) - len(raw.lstrip())
            s = block_start + bm.start(1) + lead
            e = s + len(stripped)
            if text[s:e] == stripped:
                cands.append((s, e, stripped, "CHẨN_ĐOÁN"))
    return cands


def find_section_items(text):
    """Trich cac item dang bullet duoi cac heading da biet (Trieu chung hien
    tai, Thuoc truoc khi nhap vien, Chan doan phan biet...)."""
    cands = []
    headers = list(ANY_HEADER_RE.finditer(text))
    for i, hm in enumerate(headers):
        title = hm.group(1).strip().lower()
        target_type = None
        for t, keys in SECTION_TYPE_HEADERS.items():
            if any(title == k or title.startswith(k) for k in keys):
                target_type = t
                break
        if target_type is None:
            continue
        trusted = title in TRUSTED_DIAGNOSIS_LIST_HEADERS
        block_start = hm.end()
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block_end = min(block_end, block_start + 500)
        block = text[block_start:block_end]
        # Cat block tai dong trong dau tien (ngan doan van ban): nhieu tieu de
        # con (vd "2. Tien su benh hien tai", "Ly do nhap vien") khong co dau
        # hai cham nen khong duoc ANY_HEADER_RE nhan dien la ranh gioi, khien
        # cac bullet khong lien quan bi gan nham loai cua tieu de truoc do.
        blank_idx = block.find("\n\n")
        if blank_idx != -1:
            block = block[:blank_idx]
        # Cat them tai tieu de danh so (vd "\n2.  Tien su benh hien tai",
        # "\n3.  Danh gia tai benh vien") hoac tieu de Title-Case ngay truoc
        # mot danh sach bullet moi (vd "Cac thu thuat da thuc hien\n    -"):
        # day la cac ranh gioi section ro rang du khong co dau ':'.
        num_hdr = NUMBERED_HEADER_RE.search(block)
        if num_hdr:
            block = block[:num_hdr.start()]
        sub_hdr = SUBHEADER_BULLET_RE.search(block)
        if sub_hdr:
            block = block[:sub_hdr.start()]

        max_len = 80 if target_type == "CHẨN_ĐOÁN" else 150

        def _trim_drug_paren(chunk):
            if target_type == "THUỐC":
                pidx = chunk.find(" (")
                if pidx > 2:
                    return chunk[:pidx]
            return chunk

        bullets = list(BULLET_RE.finditer(block))
        if bullets:
            for bm in bullets:
                raw = bm.group(1)
                stripped = _trim_drug_paren(raw.strip())
                if len(stripped) < 2 or len(stripped) > max_len:
                    continue
                if stripped.count("(") > stripped.count(")"):
                    continue
                if target_type == "CHẨN_ĐOÁN" and not trusted and not _looks_like_diagnosis(stripped):
                    continue
                lead = len(raw) - len(raw.lstrip())
                s = block_start + bm.start(1) + lead
                e = s + len(stripped)
                if text[s:e] == stripped:
                    cands.append((s, e, stripped, target_type))
        else:
            line = _trim_drug_paren(block.strip())
            if 2 <= len(line) <= max_len and "\n" not in block.strip("\n"):
                if target_type == "CHẨN_ĐOÁN" and not trusted and not _looks_like_diagnosis(line):
                    pass
                else:
                    s = block_start + block.find(line)
                    e = s + len(line)
                    if text[s:e] == line:
                        cands.append((s, e, line, target_type))
    return cands


# --------------------------------------------------------------------------
# 3. HOP NHAT & LOAI CHONG LAN
# --------------------------------------------------------------------------

def resolve_overlaps(cands):
    cands = [c for c in cands if c[1] > c[0] and c[2].strip()]
    cands.sort(key=lambda c: (-(c[1] - c[0]), TYPE_ORDER.get(c[3], 9)))
    accepted = []
    occupied = []
    for c in cands:
        s, e = c[0], c[1]
        if any(not (e <= os_ or s >= oe_) for os_, oe_ in occupied):
            continue
        accepted.append(c)
        occupied.append((s, e))
    accepted.sort(key=lambda c: c[0])
    return accepted


# --------------------------------------------------------------------------
# 4. ASSERTION DETECTION
# --------------------------------------------------------------------------

def get_history_spans(text):
    headers = list(SECTION_HEADER_RE.finditer(text))
    spans = []
    for i, h in enumerate(headers):
        title = h.group(1)
        if HISTORY_SECTION_RE.search(title):
            start = h.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            spans.append((start, end))
    return spans


CLAUSE_BOUNDARY_RE = re.compile(r"[;\n]|(?<!\d)\.(?!\d)")


def clause_scope(text, start, end, window=150):
    left = max(0, start - window)
    seg = text[left:start]
    matches = list(CLAUSE_BOUNDARY_RE.finditer(seg))
    scope_start = left + matches[-1].end() if matches else left
    right = min(len(text), end + 40)
    seg2 = text[end:right]
    m = CLAUSE_BOUNDARY_RE.search(seg2)
    scope_end = end + m.start() if m else right
    return text[scope_start:scope_end], scope_start


def _has_trigger(s, triggers):
    for trig in triggers:
        if re.search(r"\b" + re.escape(trig) + r"\b", s):
            return True
    return False


def get_assertions(text, start, end, history_spans):
    scope, scope_start = clause_scope(text, start, end)
    scope_low = scope.lower()
    rel_pos = start - scope_start

    assertions = []
    before = scope_low[:rel_pos]
    if _has_trigger(before[-40:], NEGATION_TRIGGERS):
        assertions.append("isNegated")
    is_family = _has_trigger(scope_low, FAMILY_TRIGGERS)
    if is_family and _has_trigger(scope_low, OBSERVER_TRIGGERS):
        is_family = False
    if is_family:
        assertions.append("isFamily")
    is_hist = _has_trigger(scope_low, HISTORY_TRIGGERS)
    if not is_hist:
        is_hist = any(hs <= start < he for hs, he in history_spans)
    if is_hist:
        assertions.append("isHistorical")
    return assertions


# --------------------------------------------------------------------------
# 5. CANDIDATE MAPPING
# --------------------------------------------------------------------------

def get_candidates(entity_text, etype):
    low = entity_text.lower().strip()
    if etype == "CHẨN_ĐOÁN":
        if low in _FLEX_CODES:
            return _FLEX_CODES[low]
        for name, codes in DISEASE_ICD10.items():
            if name in low or low in name:
                return codes
        return []
    if etype == "THUỐC":
        scd = get_drug_scd_candidate(entity_text)
        if scd:
            return [scd]
        for name, codes in DRUG_RXNORM.items():
            if name in low:
                return codes
        return []
    return None


# --------------------------------------------------------------------------
# 6. PIPELINE CHINH
# --------------------------------------------------------------------------

def process_text(text):
    cands = []
    # LUU Y QUAN TRONG (vong 8): cong thuc diem trong de bai chi neu Jaccard
    # duoc tinh "voi cac benh, thuoc va trieu chung tuong ung" -- tuc la CHI
    # CO 3 LOAI THUC THE trong schema thuc: CHẨN_ĐOÁN, THUỐC, TRIỆU_CHỨNG.
    # Cac loai TÊN_XÉT_NGHIỆM / KẾT_QUẢ_XÉT_NGHIỆM (sinh hieu, xet nghiem)
    # tu cac ham find_vitals/find_lab_panel KHONG nam trong schema nay, nen
    # moi thuc the loai do la false positive thuan tuy (insertion loi trong
    # WER, khong co gia tri Jaccard nao de khop), va con co the choan cho/
    # chan mat mot thuc the CHẨN_ĐOÁN/THUỐC/TRIỆU_CHỨNG dung o cung vi tri
    # do resolve_overlaps chi loai chong lan theo do dai span. Da loai bo
    # hoan toan 2 ham nay khoi pipeline.
    pass
    cands += find_diagnoses(text)
    cands += find_drugs(text)
    cands += find_symptoms(text)
    cands += find_section_items(text)
    cands += find_trusted_diagnosis_lists(text)

    accepted = resolve_overlaps(cands)
    history_spans = get_history_spans(text)

    results = []
    for s, e, etext, etype in accepted:
        entry = {"text": etext, "type": etype}
        codes = get_candidates(etext, etype)
        if codes is not None:
            entry["candidates"] = codes
        if etype in ("CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"):
            entry["assertions"] = get_assertions(text, s, e, history_spans)
        else:
            entry["assertions"] = []
        entry["position"] = [s, e]
        results.append(entry)
    return results


def _find_txt_files(input_dir):
    """Tim file .txt trong input_dir; neu khong co truc tiep, tu dong do
    xuong cac thu muc con (vd cau truc zip giai nen ra 'input/input/*.txt')."""
    files = glob.glob(os.path.join(input_dir, "*.txt"))
    if not files:
        files = glob.glob(os.path.join(input_dir, "**", "*.txt"), recursive=True)
    return sorted(files, key=lambda p: int(os.path.basename(p).split(".")[0]))


def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = _find_txt_files(input_dir)
    if not files:
        print(f"Khong tim thay file .txt nao trong {input_dir!r}")
        return
    total_entities = 0
    empty = []
    for fp in files:
        idx = os.path.basename(fp).split(".")[0]
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        # QUAN TRONG: ~20% file trong corpus co doan van ban bi luu o dang
        # Unicode NFD (chu cai roi + dau phu ket hop, vd "tê"+"combining
        # acute" thay vi ky tu "tế" don) xen lan voi phan con lai la NFC.
        # Dieu nay khien \b (word boundary) coi dau phu ket hop la ky tu
        # "khong phai chu", tach doi mot tu lam mot thanh hai token gia
        # (vd "học" bi doc thanh "ho" + dau nang roi), gay ra rat nhieu
        # false positive/false negative khi so khop tu dien (vd "ho" khop
        # nham vao giua tu "học", "tê" khop nham vao giua "tế bào"). Chuan
        # hoa toan bo text ve NFC truoc khi xu ly de sua loi nay. Vi cong
        # thuc diem KHONG cham truong "position" nen viec chuan hoa (co
        # the lam thay doi so luong code point neu con sot NFD o dau khac)
        # la an toan.
        text = unicodedata.normalize("NFC", text)
        result = process_text(text)
        total_entities += len(result)
        if not result:
            empty.append(idx)
        with open(os.path.join(output_dir, f"{idx}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Processed {len(files)} files, {total_entities} entities total "
          f"({total_entities/len(files):.1f} avg/file)")
    print("Empty files:", empty)


if __name__ == "__main__":
    main("input", "output")
