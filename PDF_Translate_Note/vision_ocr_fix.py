# -*- coding: utf-8 -*-
"""
Google Vision OCR 개선 - DOCUMENT_TEXT_DETECTION 사용
문제: TEXT_DETECTION은 개별 단어 분리 → 번역 시 겹침 발생
해결: DOCUMENT_TEXT_DETECTION으로 줄 단위 그룹화
"""

def ocr_with_google_vision_v2(image_path):
    """Google Cloud Vision API로 OCR 수행 (개선 버전)
    
    변경사항:
    - text_detection → document_text_detection
    - 줄(LINE) 단위로 텍스트 그룹화
    - "행거루프" 같은 복합어가 분리되지 않음

    Returns:
        list: PaddleOCR과 동일한 형식 [[bbox, (text, confidence)], ...]
              bbox = [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    """
    client = get_vision_client()
    if client is None:
        print("[Vision OCR] Client not available, falling back to PaddleOCR")
        return None

    start_time = time.time()

    with open(image_path, 'rb') as f:
        content = f.read()

    image = vision.Image(content=content)
    
    # ★ 핵심 변경: text_detection → document_text_detection
    response = client.document_text_detection(image=image)

    if response.error.message:
        print(f"[Vision OCR] Error: {response.error.message}")
        return None

    if not response.full_text_annotation:
        print("[Vision OCR] No text detected")
        return []

    # DOCUMENT_TEXT_DETECTION 결과 파싱
    # 구조: pages → blocks → paragraphs → words → symbols
    results = []
    
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in page.blocks:
                # ★ LINE(줄) 단위로 텍스트 그룹화
                # 같은 줄에 있는 단어들을 하나로 병합
                for line_words in get_lines_from_paragraph(paragraph):
                    if not line_words:
                        continue
                    
                    # 줄의 모든 단어 텍스트 합치기
                    line_text = ''.join([word['text'] for word in line_words])
                    
                    # 줄 전체의 bounding box 계산
                    all_x = []
                    all_y = []
                    for word in line_words:
                        for vertex in word['vertices']:
                            all_x.append(vertex.x)
                            all_y.append(vertex.y)
                    
                    x1, x2 = min(all_x), max(all_x)
                    y1, y2 = min(all_y), max(all_y)
                    
                    bbox = [
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]
                    ]
                    
                    # 평균 confidence 계산
                    avg_confidence = sum(w['confidence'] for w in line_words) / len(line_words)
                    
                    results.append([bbox, (line_text, avg_confidence)])

    elapsed = time.time() - start_time
    print(f"[Vision OCR v2] Detected {len(results)} text lines in {elapsed:.2f}s")

    return results


def get_lines_from_paragraph(paragraph):
    """단락에서 줄 단위로 단어들을 그룹화
    
    같은 Y 좌표 범위에 있는 단어들을 같은 줄로 인식
    """
    words_data = []
    
    for word in paragraph.words:
        # 단어 텍스트 추출
        word_text = ''.join([symbol.text for symbol in word.symbols])
        
        # 단어 bbox 추출
        vertices = word.bounding_box.vertices
        y_center = sum(v.y for v in vertices) / 4
        
        words_data.append({
            'text': word_text,
            'vertices': vertices,
            'y_center': y_center,
            'confidence': word.confidence if hasattr(word, 'confidence') else 0.99
        })
    
    if not words_data:
        return []
    
    # Y 좌표로 정렬
    words_data.sort(key=lambda w: w['y_center'])
    
    # 같은 줄로 그룹화 (Y 좌표 차이가 글자 높이의 50% 이내)
    lines = []
    current_line = [words_data[0]]
    
    for word in words_data[1:]:
        # 이전 단어와 Y 좌표 비교
        prev_y = current_line[-1]['y_center']
        curr_y = word['y_center']
        
        # 글자 높이 추정 (현재 단어의 bbox 높이)
        word_height = max(v.y for v in word['vertices']) - min(v.y for v in word['vertices'])
        threshold = max(word_height * 0.5, 10)  # 최소 10px
        
        if abs(curr_y - prev_y) < threshold:
            # 같은 줄
            current_line.append(word)
        else:
            # 새 줄
            # X 좌표로 정렬 후 줄 추가
            current_line.sort(key=lambda w: min(v.x for v in w['vertices']))
            lines.append(current_line)
            current_line = [word]
    
    # 마지막 줄 추가
    if current_line:
        current_line.sort(key=lambda w: min(v.x for v in w['vertices']))
        lines.append(current_line)
    
    return lines


# ============================================================
# 더 간단한 대안: 단어 병합 알고리즘 (기존 text_detection 사용 시)
# ============================================================

def merge_adjacent_texts(texts, horizontal_threshold=5, vertical_threshold=10):
    """인접한 텍스트를 병합
    
    Args:
        texts: [[bbox, (text, confidence)], ...] 형식의 OCR 결과
        horizontal_threshold: 수평 방향 병합 임계값 (픽셀)
        vertical_threshold: 수직 방향 병합 임계값 (픽셀)
    
    Returns:
        병합된 텍스트 리스트
    """
    if not texts:
        return []
    
    # bbox에서 좌표 추출 헬퍼
    def get_coords(bbox):
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)
    
    # 병합 여부 확인
    def should_merge(bbox1, bbox2):
        x1_min, y1_min, x1_max, y1_max = get_coords(bbox1)
        x2_min, y2_min, x2_max, y2_max = get_coords(bbox2)
        
        # Y 좌표가 비슷해야 함 (같은 줄)
        y1_center = (y1_min + y1_max) / 2
        y2_center = (y2_min + y2_max) / 2
        
        if abs(y1_center - y2_center) > vertical_threshold:
            return False
        
        # X 좌표가 인접해야 함
        # bbox1의 오른쪽과 bbox2의 왼쪽 거리
        gap = x2_min - x1_max
        
        return -5 <= gap <= horizontal_threshold  # 약간 겹치거나 인접
    
    # 결과를 리스트로 변환 (수정 가능하게)
    result = [list(t) for t in texts]
    
    # 반복적으로 병합
    merged = True
    while merged:
        merged = False
        new_result = []
        used = set()
        
        for i, item1 in enumerate(result):
            if i in used:
                continue
            
            bbox1, (text1, conf1) = item1
            merged_bbox = bbox1
            merged_text = text1
            merged_conf = conf1
            count = 1
            
            for j, item2 in enumerate(result):
                if i >= j or j in used:
                    continue
                
                bbox2, (text2, conf2) = item2
                
                if should_merge(merged_bbox, bbox2):
                    # 병합
                    used.add(j)
                    merged = True
                    
                    # 새 bbox 계산
                    x1_min, y1_min, x1_max, y1_max = get_coords(merged_bbox)
                    x2_min, y2_min, x2_max, y2_max = get_coords(bbox2)
                    
                    new_x_min = min(x1_min, x2_min)
                    new_y_min = min(y1_min, y2_min)
                    new_x_max = max(x1_max, x2_max)
                    new_y_max = max(y1_max, y2_max)
                    
                    merged_bbox = [
                        [new_x_min, new_y_min],
                        [new_x_max, new_y_min],
                        [new_x_max, new_y_max],
                        [new_x_min, new_y_max]
                    ]
                    
                    # 텍스트 병합 (왼쪽에서 오른쪽 순서로)
                    if x1_min <= x2_min:
                        merged_text = merged_text + text2
                    else:
                        merged_text = text2 + merged_text
                    
                    merged_conf = (merged_conf * count + conf2) / (count + 1)
                    count += 1
            
            new_result.append([merged_bbox, (merged_text, merged_conf)])
        
        result = new_result
    
    return result
