import time
import re
from datetime import datetime

import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from googleapiclient.discovery import build
from google.oauth2 import service_account

# --- [사용자 설정 영역] ---
# 우선순위: st.secrets (Streamlit Cloud) > .env (로컬)
load_dotenv()

def get_secret(key, default=None):
    """st.secrets 우선, 없으면 환경변수(.env) 참조"""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key, default)

GEMINI_API_KEY = get_secret('GEMINI_API_KEY')
SPREADSHEET_ID = get_secret('SPREADSHEET_ID')
SHEET_NAME_PRO_KOR = get_secret('SHEET_NAME_PRO_KOR', 'Pro_Kor')
SHEET_NAME_PRO_ENG = get_secret('SHEET_NAME_PRO_ENG', 'Pro_Eng')
MODEL_NAME = get_secret('MODEL_NAME', 'models/gemini-2.0-flash')

# --- 상수 ---
SUBJECT_LIST = [
    'Databricks Tooling',
    'Data Processing',
    'Data Modeling',
    'Security and Governance',
    'Monitoring and Logging',
    'Testing and Deployment',
]

SUBJECT_KEYWORDS = {
    'Databricks Tooling':       ['tooling'],
    'Data Processing':          ['processing'],
    'Data Modeling':            ['modeling', 'modelling'],
    'Security and Governance':  ['security', 'governance'],
    'Monitoring and Logging':   ['monitoring', 'logging'],
    'Testing and Deployment':   ['testing', 'deployment'],
}

PROMPT_TEMPLATE = """당신은 Databricks Certified Data Engineer Professional 시험 문제 분류 전문가입니다.

아래 시험 문제를 분석하여 세 가지를 출력하세요.

1) subject: 아래 6개 중 정확히 하나
   - Databricks Tooling (Cluster, Jobs, Notebook, Repos, Workspace, API, CLI, DBFS, Widget, Scheduling)
   - Data Processing (Spark SQL, DataFrame, Structured Streaming, Auto Loader, COPY INTO, DLT Pipeline, CDC, Batch/Stream ETL)
   - Data Modeling (Delta Lake, Table Design, Schema, Partitioning, Z-order, OPTIMIZE, VACUUM, Medalion Architecture)
   - Security and Governance (Unity Catalog, ACL, Priviledge, Data Masking, Encryption, Audit, Service Principal)
   - Monitoring and Logging (Query Monitoring, Alert, Dashboard, Logging, Performance Metric, Spark UI, Event Log)
   - Testing and Deployment (CI/CD, Test, Deployement, Promotion, Environment Management, Multiple workspaces, Version Management)

2) category: 해당 문제에서 가장 핵심적인 키워드 1개를 반드시 영문으로 출력 (예: "Auto Loader", "VACUUM", "Unity Catalog", "DLT", "Structured Streaming")

3) title: 해당 문제의 구체적 세부 주제를 한글로 간결하게 추론하세요 (예: "위젯 파라미터 전달", "클러스터 권한 관리", "Auto Loader 스키마 진화")

--- 문제 ---
문제번호: {q_num}
문제: {q_text}
보기: {q_choices}
참고: {q_ref}
--- 끝 ---

반드시 아래 형식 3줄로만 출력하세요. 마크다운이나 기호 없이 텍스트만:
subject: 카테고리명
category: 핵심키워드
title: 세부주제"""

FALLBACK_PROMPT_TEMPLATE = """Databricks DE Professional 시험 문제 분류:
문제 키워드: {q_summary}

subject(6개 중 택1): Databricks Tooling / Data Processing / Data Modeling / Security and Governance / Monitoring and Logging / Testing and Deployment
category: 문제의 가장 핵심적인 키워드 1개 (반드시 영문)
title: 세부주제를 한글로

형식:
subject: 카테고리명
category: 핵심키워드
title: 세부주제"""

DESC_PROMPT_TEMPLATE = """당신은 Databricks Certified Data Engineer Professional 시험 해설 전문가입니다.

아래 시험 문제와 보기를 분석하여, 이 문제에 대한 간결하고 명확한 해설을 작성하세요.

--- 문제 ---
문제번호: {q_num}
문제: {q_text}
보기: {q_choices}
--- 끝 ---

다음 내용을 포함하여 해설을 작성하세요:
1) 이 문제가 다루는 핵심 개념을 1-2문장으로 설명
2) 정답과 그 이유
3) 주요 오답이 틀린 이유 (간결하게)

마크다운 없이 텍스트만 출력하세요. 전체 5-8문장 이내로 간결하게 작성하세요."""

SAFETY = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# 번역이 필요한 컬럼 (한↔영)
TRANSLATE_COLS = {'title', 'q_text', 'options', 'desc', 'memo'}
# 번역 불필요, 그대로 복사
COPY_COLS = {'subject', 'category'}


# --- 핵심 함수 ---
def match_subject(text):
    t = text.lower()
    for subj in SUBJECT_LIST:
        if subj.lower() in t:
            return subj
    for subj, kws in SUBJECT_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                return subj
    return None


def parse_response(text):
    subject = None
    category = None
    title = None
    cleaned = re.sub(r'[*`#]', '', text)
    for line in cleaned.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        line_lower = line.lower()
        if line_lower.startswith('subject'):
            val = re.split(r':\s*', line, maxsplit=1)
            if len(val) >= 2:
                subject = match_subject(val[1].strip())
        elif line_lower.startswith('category'):
            val = re.split(r':\s*', line, maxsplit=1)
            if len(val) >= 2:
                category = val[1].strip()
        elif line_lower.startswith('title'):
            val = re.split(r':\s*', line, maxsplit=1)
            if len(val) >= 2:
                title = val[1].strip()
    if not subject:
        subject = match_subject(cleaned)
    return subject, category, title


def get_sheets_service():
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    # 우선순위: st.secrets > .env 파일 경로
    try:
        info = dict(st.secrets['GCP_SERVICE_ACCOUNT'])
        creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    except (KeyError, FileNotFoundError):
        sa_file = os.environ.get('GCP_SERVICE_ACCOUNT_FILE', 'mydatabyai-42c0d2826e21.json')
        creds = service_account.Credentials.from_service_account_file(sa_file, scopes=SCOPES)
    return build('sheets', 'v4', credentials=creds)


def col_idx_to_letter(idx):
    """0-based column index -> 열 문자 (0->A, 1->B, ...)"""
    return chr(ord('A') + idx)


def update_range(sheets_service, sheet_name, row_num, col_start, col_end, values):
    """시트의 특정 행 범위에 값 업데이트"""
    sheets_service.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=f"'{sheet_name}'!{col_start}{row_num}:{col_end}{row_num}",
        valueInputOption='RAW',
        body={'values': [values]}
    ).execute()


def update_single_cell(sheets_service, sheet_name, row_num, col_letter, value):
    """시트의 특정 셀 1개 업데이트"""
    sheets_service.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=f"'{sheet_name}'!{col_letter}{row_num}",
        valueInputOption='RAW',
        body={'values': [[value]]}
    ).execute()


def gemini_call(model, prompt, max_tokens=500):
    """Gemini API 호출 (공통)"""
    try:
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.0,
            ),
            safety_settings=SAFETY,
        )
        if not resp.candidates:
            return None, '응답 없음'
        candidate = resp.candidates[0]
        if candidate.finish_reason and candidate.finish_reason.value == 2:
            return None, 'SAFETY_BLOCKED'
        return resp.text, ''
    except Exception as e:
        return None, f'{type(e).__name__}: {str(e)[:200]}'


def is_korean(text):
    """텍스트에 한글이 포함되어 있는지 판별"""
    if not text:
        return False
    korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7a3' or '\u3131' <= c <= '\u3163')
    return korean_chars > len(text) * 0.05


def translate_text(model, text, direction='kor_to_eng'):
    """Gemini를 사용하여 텍스트 번역"""
    if not text or not text.strip():
        return ''
    if direction == 'kor_to_eng':
        prompt = f"Translate the following Korean text to English accurately. Output ONLY the translated text, nothing else.\n\n{text}"
    else:
        prompt = f"다음 영어 텍스트를 한국어로 정확하게 번역하세요. 번역된 텍스트만 출력하세요.\n\n{text}"
    result, err = gemini_call(model, prompt, max_tokens=1000)
    return result.strip() if result else text


def generate_desc(model, q_num, q_text, q_choices):
    """문제 해설 생성"""
    prompt = DESC_PROMPT_TEMPLATE.format(
        q_num=q_num,
        q_text=q_text[:500],
        q_choices=q_choices[:500],
    )
    result, err = gemini_call(model, prompt, max_tokens=1000)
    return (result.strip() if result else ''), err


def classify_row(model, q_num, q_text, q_choices, q_ref):
    prompt = PROMPT_TEMPLATE.format(
        q_num=q_num,
        q_text=q_text[:400],
        q_choices=q_choices[:400],
        q_ref=q_ref[:300]
    )
    last_error = ''
    for attempt in range(3):
        try:
            use_prompt = prompt
            if 'SAFETY_BLOCKED' in last_error:
                q_summary = q_text[:150].replace('\n', ' ')
                use_prompt = FALLBACK_PROMPT_TEMPLATE.format(q_summary=q_summary)

            resp = model.generate_content(
                use_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=100,
                    temperature=0.0,
                ),
                safety_settings=SAFETY,
            )
            if not resp.candidates:
                last_error = f'[시도{attempt+1}] 응답 없음 (candidates 비어있음)'
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                continue
            candidate = resp.candidates[0]
            finish_reason = candidate.finish_reason
            if finish_reason and finish_reason.value == 2:
                last_error = f'[시도{attempt+1}] SAFETY_BLOCKED'
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                continue
            raw_text = resp.text
            subject, category, title = parse_response(raw_text)
            if subject:
                return subject, category or '', title or '', ''
            else:
                last_error = f'[시도{attempt+1}] 파싱 실패 | 응답원문: {raw_text.strip()[:200]}'
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
        except Exception as e:
            last_error = f'[시도{attempt+1}] {type(e).__name__}: {str(e)[:200]}'
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    return None, None, None, last_error


def get_val(row, idx):
    """행에서 특정 인덱스 값을 안전하게 가져오기"""
    return row[idx].strip() if len(row) > idx and row[idx].strip() else ''


def read_sheet_data(sheets_service, sheet_name):
    """시트 데이터 전체 읽기 (헤더 + 데이터)"""
    try:
        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"'{sheet_name}'!A:J"
        ).execute()
        all_rows = result.get('values', [])
        if not all_rows:
            return [], []
        return all_rows[0], all_rows[1:]
    except Exception:
        return [], []


def build_qno_map(data_rows, idx_qnum):
    """q_no 기준으로 행 매핑 {번호: (sheet_row, row_data)}"""
    qno_map = {}
    for idx, row in enumerate(data_rows):
        sheet_row = idx + 2 # Google Sheet rows are 1-based, and we skip header
        q_num_str = row[idx_qnum] if len(row) > idx_qnum else ''
        try:
            num = int(q_num_str.replace('Q.', '').strip())
            qno_map[num] = (sheet_row, row)
        except Exception:
            continue
    return qno_map

def get_max_q_no(sheets_service):
    """Pro_Kor, Pro_Eng 시트에서 가장 큰 q_no를 찾아 반환"""
    max_q_no = 0
    # Pro_Kor 시트에서 최대 q_no 찾기
    kor_header, kor_data_rows = read_sheet_data(sheets_service, SHEET_NAME_PRO_KOR)
    if kor_header:
        COL_KOR = {h.strip().replace('\\n', ''): i for i, h in enumerate(kor_header)}
        IDX_QNUM_KOR = COL_KOR.get('q_no', -1)
        if IDX_QNUM_KOR != -1:
            for row in kor_data_rows:
                q_num_str = get_val(row, IDX_QNUM_KOR)
                try:
                    num = int(q_num_str.replace('Q.', '').strip())
                    if num > max_q_no:
                        max_q_no = num
                except ValueError:
                    pass
    
    # Pro_Eng 시트에서 최대 q_no 찾기
    eng_header, eng_data_rows = read_sheet_data(sheets_service, SHEET_NAME_PRO_ENG)
    if eng_header:
        COL_ENG = {h.strip().replace('\\n', ''): i for i, h in enumerate(eng_header)}
        IDX_QNUM_ENG = COL_ENG.get('q_no', -1)
        if IDX_QNUM_ENG != -1:
            for row in eng_data_rows:
                q_num_str = get_val(row, IDX_QNUM_ENG)
                try:
                    num = int(q_num_str.replace('Q.', '').strip())
                    if num > max_q_no:
                        max_q_no = num
                except ValueError:
                    pass
    return max_q_no


def append_row_to_sheet(sheets_service, sheet_name, headers, row_dict):
    """시트에 한 행 추가 (row_dict: {컬럼명: 값})"""
    header_map = {h.strip().replace('\\n', ''): i for i, h in enumerate(headers)}
    new_row_values = [''] * len(headers)
    for col_name, value in row_dict.items():
        if col_name in header_map:
            new_row_values[header_map[col_name]] = value
    sheets_service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=f"'{sheet_name}'!A:A",
        valueInputOption='RAW',
        insertDataOption='INSERT_ROWS',
        body={'values': [new_row_values]}
    ).execute()


def add_new_question_synced(model, sheets_service, q_no_int, q_text, q_options,
                            kor_headers, eng_headers):
    """새 문제를 Pro_Kor, Pro_Eng 양쪽에 동기화하여 추가.
    입력 언어를 자동 감지하여 각 시트에 맞는 언어로 저장."""
    q_num_str = f"Q.{q_no_int:03d}"
    input_is_korean = is_korean(q_text)

    # --- 1) AI 분류 (subject, category, title) ---
    status = st.empty()
    status.text(f'AI 분류 중: {q_num_str}...')
    subject, category, title, err = classify_row(model, q_num_str, q_text, q_options, '')
    if not subject:
        st.error(f'AI 분류 실패: {err}')
        return
    time.sleep(1)

    # --- 2) AI 해설 생성 ---
    status.text(f'해설 생성 중: {q_num_str}...')
    desc, desc_err = generate_desc(model, q_num_str, q_text, q_options)
    time.sleep(1)

    # --- 3) 언어별 데이터 준비 ---
    status.text(f'번역 중: {q_num_str}...')
    if input_is_korean:
        kor_q_text = q_text
        kor_options = q_options
        kor_title = title  # classify_row의 title은 한글
        kor_desc = desc    # desc도 한글
        eng_q_text = translate_text(model, q_text, 'kor_to_eng'); time.sleep(1)
        eng_options = translate_text(model, q_options, 'kor_to_eng'); time.sleep(1)
        eng_title = translate_text(model, title, 'kor_to_eng') if title else ''; time.sleep(1)
        eng_desc = translate_text(model, desc, 'kor_to_eng') if desc else ''
    else:
        eng_q_text = q_text
        eng_options = q_options
        # 영문 입력이면 title/desc도 영문으로 다시 생성
        eng_title = translate_text(model, title, 'kor_to_eng') if title else ''; time.sleep(1)
        eng_desc = translate_text(model, desc, 'kor_to_eng') if desc else ''; time.sleep(1)
        kor_q_text = translate_text(model, q_text, 'eng_to_kor'); time.sleep(1)
        kor_options = translate_text(model, q_options, 'eng_to_kor'); time.sleep(1)
        kor_title = title  # 이미 한글
        kor_desc = desc    # 이미 한글

    # --- 4) 양쪽 시트에 추가 ---
    status.text(f'시트 저장 중: {q_num_str}...')
    kor_row = {
        'q_no': q_num_str, 'q_text': kor_q_text, 'options': kor_options,
        'subject': subject, 'category': category,
        'title': kor_title, 'desc': kor_desc,
    }
    eng_row = {
        'q_no': q_num_str, 'q_text': eng_q_text, 'options': eng_options,
        'subject': subject, 'category': category,
        'title': eng_title, 'desc': eng_desc,
    }

    if kor_headers:
        append_row_to_sheet(sheets_service, SHEET_NAME_PRO_KOR, kor_headers, kor_row)
    if eng_headers:
        append_row_to_sheet(sheets_service, SHEET_NAME_PRO_ENG, eng_headers, eng_row)

    status.empty()
    st.success(f"Q.{q_no_int:03d} → Pro_Kor(한글) + Pro_Eng(영문) 양쪽 시트에 추가 완료!")


def sync_row(model, sheets_service, col_map, kor_row, eng_row, kor_sheet_row, eng_sheet_row, log_lines):
    """Pro_Kor ↔ Pro_Eng 양방향 동기화 (빈 셀만 채움)"""
    if not eng_sheet_row:
        return

    for col_name in TRANSLATE_COLS | COPY_COLS:
        if col_name not in col_map:
            continue
        idx = col_map[col_name]
        kor_val = get_val(kor_row, idx) if kor_row else ''
        eng_val = get_val(eng_row, idx) if eng_row else ''
        col_letter = col_idx_to_letter(idx)

        if kor_val and not eng_val:
            if col_name in TRANSLATE_COLS:
                new_val = translate_text(model, kor_val, 'kor_to_eng')
                time.sleep(1)
            else:
                new_val = kor_val
            update_single_cell(sheets_service, SHEET_NAME_PRO_ENG, eng_sheet_row, col_letter, new_val)
            log_lines.append(f'  SYNC {col_name}: KOR->ENG')

        elif eng_val and not kor_val:
            if col_name in TRANSLATE_COLS:
                new_val = translate_text(model, eng_val, 'eng_to_kor')
                time.sleep(1)
            else:
                new_val = eng_val
            update_single_cell(sheets_service, SHEET_NAME_PRO_KOR, kor_sheet_row, col_letter, new_val)
            log_lines.append(f'  SYNC {col_name}: ENG->KOR')


# --- Streamlit UI ---
st.set_page_config(page_title='DBX Pro 문제 분류', layout='wide')

# 사이드바 메뉴
MENU = {
    'Databricks Pro 문제은행 업데이트': '🏷️',
}
with st.sidebar:
    st.header('DBX Pro')
    selected_menu = st.radio('메뉴', list(MENU.keys()), format_func=lambda x: f'{MENU[x]} {x}')
    st.divider()
    st.link_button(
        '📊 Google Spreadsheet 열기',
        'https://docs.google.com/spreadsheets/d/1hcMfygRCxmgADm0Vf0Fbr8gANXPyNbivhTkMNel9MM0/edit?gid=1358331458#gid=1358331458',
        use_container_width=True,
    )

# --- 페이지: Databricks Pro 문제은행 업데이트 ---
if selected_menu == 'Databricks Pro 문제은행 업데이트':
    st.title('Databricks Pro 문제은행 업데이트')

    # New main columns for the entire page content
    main_col_left, main_col_right = st.columns([0.5, 0.5]) # Adjust ratio as needed for the two main sections

    with main_col_left:
        st.subheader('기존 문제 업데이트')
        st.markdown("---") # Visual separator
        col1, col2 = st.columns(2)
        with col1:
            start_question_number = st.number_input(
                '시작번호', min_value=1, max_value=999, value=1, step=1, key='update_start_q'
            )
        with col2:
            end_question_number = st.number_input(
                '종료번호', min_value=1, max_value=999, value=111, step=1, key='update_end_q'
            )

        st.caption('Overwrite (체크 시 기존 값이 있어도 덮어쓰기)')
        ow1, ow2, ow3, ow4 = st.columns(4)
        with ow1:
            ow_subject = st.checkbox('subject', key='ow_subject')
        with ow2:
            ow_category = st.checkbox('category', key='ow_category')
        with ow3:
            ow_title = st.checkbox('title', key='ow_title')
        with ow4:
            ow_desc = st.checkbox('desc', key='ow_desc')

        if st.button('시작', type='primary', use_container_width=True, key='start_update_button'):
            if start_question_number > end_question_number:
                st.error('시작번호가 종료번호보다 큽니다.')
            else:
                # 초기화
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel(MODEL_NAME)
                sheets_service = get_sheets_service()

                # 양쪽 시트 데이터 읽기
                kor_header, kor_data_rows = read_sheet_data(sheets_service, SHEET_NAME_PRO_KOR)
                eng_header, eng_data_rows = read_sheet_data(sheets_service, SHEET_NAME_PRO_ENG)

                if not kor_header:
                    st.error('Pro_Kor 시트를 읽을 수 없습니다.')
                else:
                    has_eng = len(eng_header) > 0

                    COL = {h.strip().replace('\n', ''): i for i, h in enumerate(kor_header)}
                    IDX_SUBJ    = COL.get('subject', 0)
                    IDX_CAT     = COL.get('category', 1)
                    IDX_TITLE   = COL.get('title', 2)
                    IDX_QNUM    = COL.get('q_no', 3)
                    IDX_QTEXT   = COL.get('q_text', 4)
                    IDX_CHOICES = COL.get('options', 5)
                    IDX_DESC    = COL.get('desc', 6)

                    # q_no 기준 매핑
                    kor_qno_map = build_qno_map(kor_data_rows, IDX_QNUM)
                    eng_qno_map = build_qno_map(eng_data_rows, IDX_QNUM) if has_eng else {}

                    # 대상 행 필터링
                    target_nums = sorted(
                        n for n in kor_qno_map
                        if start_question_number <= n <= end_question_number
                    )

                    total = len(target_nums)
                    if total == 0:
                        st.warning('대상 문항이 없습니다.')
                    else:
                        success = 0
                        fail = 0
                        skip = 0
                        fail_list = []
                        start_time = datetime.now()

                        st.info(
                            f'대상 범위: Q.{start_question_number:03d} ~ Q.{end_question_number:03d} '
                            f'({total}문항) | 모델: {MODEL_NAME}'
                        )

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        log_area = st.empty()
                        log_lines = []

                        for i, num in enumerate(target_nums, 1):
                            q_label = f'Q.{num:03d}'

                            kor_sheet_row, kor_row = kor_qno_map[num]
                            eng_sheet_row, eng_row = eng_qno_map.get(num, (None, []))

                            # 기존 값 확인 (한글 시트)
                            existing_subj = get_val(kor_row, IDX_SUBJ)
                            existing_cat  = get_val(kor_row, IDX_CAT)
                            existing_ttl  = get_val(kor_row, IDX_TITLE)
                            existing_desc = get_val(kor_row, IDX_DESC)

                            # 헤더행 제외
                            is_header = existing_subj == 'subject'

                            # 분류 필요 여부
                            need_subj = not existing_subj or is_header or ow_subject
                            need_cat  = not existing_cat  or is_header or ow_category
                            need_ttl  = not existing_ttl  or is_header or ow_title
                            need_classify = need_subj or need_cat or need_ttl

                            # desc 생성 여부 (체크 시에만 작동)
                            need_desc = ow_desc

                            # 영문 시트 동기화 필요 여부 (없는 행이면 새로 추가)
                            need_sync = has_eng

                            if not need_classify and not need_desc and not need_sync:
                                skip += 1
                                log_lines.append(f'[{i:03d}/{total}] {q_label} -> SKIP ({existing_subj})')
                                log_area.code('\n'.join(log_lines[-30:]))
                                progress_bar.progress(i / total)
                                continue

                            status_text.text(f'처리 중: {q_label} ({i}/{total})')

                            q_text    = get_val(kor_row, IDX_QTEXT)
                            q_choices = get_val(kor_row, IDX_CHOICES)
                            q_desc    = get_val(kor_row, IDX_DESC)

                            # --- 1) 분류 ---
                            if need_classify:
                                subject, category, title, err_msg = classify_row(
                                    model, q_label, q_text, q_choices, q_desc
                                )

                                if subject:
                                    final_subj = subject        if need_subj else existing_subj
                                    final_cat  = category or '' if need_cat  else existing_cat
                                    final_ttl  = title or ''    if need_ttl  else existing_ttl

                                    # 한글 시트 업데이트
                                    update_range(sheets_service, SHEET_NAME_PRO_KOR,
                                                 kor_sheet_row, 'A', 'C',
                                                 [final_subj, final_cat, final_ttl])

                                    # 영문 시트 업데이트 (없으면 새 행 추가)
                                    eng_ttl = translate_text(model, final_ttl, 'kor_to_eng') if final_ttl else ''
                                    time.sleep(1)
                                    if eng_sheet_row:
                                        update_range(sheets_service, SHEET_NAME_PRO_ENG,
                                                     eng_sheet_row, 'A', 'C',
                                                     [final_subj, final_cat, eng_ttl])
                                    elif has_eng:
                                        # Pro_Eng에 해당 문제가 없으면 번역하여 새 행 추가
                                        eng_q_text = translate_text(model, q_text, 'kor_to_eng'); time.sleep(1)
                                        eng_choices = translate_text(model, q_choices, 'kor_to_eng'); time.sleep(1)
                                        eng_desc = translate_text(model, q_desc, 'kor_to_eng') if q_desc else ''; time.sleep(1)
                                        eng_row_dict = {
                                            'q_no': q_label, 'q_text': eng_q_text, 'options': eng_choices,
                                            'subject': final_subj, 'category': final_cat,
                                            'title': eng_ttl, 'desc': eng_desc,
                                        }
                                        append_row_to_sheet(sheets_service, SHEET_NAME_PRO_ENG, eng_header, eng_row_dict)
                                        log_lines.append(f'  Pro_Eng에 새 행 추가')

                                    success += 1
                                    log_lines.append(
                                        f'[{i:03d}/{total}] {q_label} -> {final_subj} | {final_cat} | {final_ttl}'
                                    )
                                else:
                                    fail += 1
                                    fail_list.append(q_label)
                                    log_lines.append(
                                        f'[{i:03d}/{total}] {q_label} -> FAIL | {err_msg}'
                                    )
                                    log_area.code('\n'.join(log_lines[-30:]))
                                    progress_bar.progress(i / total)
                                    if i < total:
                                        time.sleep(4)
                                    continue
                            else:
                                log_lines.append(f'[{i:03d}/{total}] {q_label} -> OK ({existing_subj})')

                            # --- 2) desc 생성 ---
                            if need_desc:
                                status_text.text(f'해설 생성 중: {q_label} ({i}/{total})')
                                desc_text, desc_err = generate_desc(model, q_label, q_text, q_choices)
                                if desc_text:
                                    desc_col = col_idx_to_letter(IDX_DESC)
                                    update_single_cell(sheets_service, SHEET_NAME_PRO_KOR,
                                                       kor_sheet_row, desc_col, desc_text)
                                    # 영문 시트에도 번역하여 저장
                                    if eng_sheet_row:
                                        time.sleep(1)
                                        desc_eng = translate_text(model, desc_text, 'kor_to_eng')
                                        update_single_cell(sheets_service, SHEET_NAME_PRO_ENG,
                                                           eng_sheet_row, desc_col, desc_eng)
                                    # eng_sheet_row가 없는 경우는 위 분류 단계에서 이미 새 행을 추가했으므로 skip
                                    log_lines.append(f'  desc 생성 완료')
                                else:
                                    log_lines.append(f'  desc 생성 실패: {desc_err}')
                                time.sleep(1)

                            # --- 3) 양방향 동기화 ---
                            if need_sync:
                                sync_row(model, sheets_service, COL,
                                         kor_row, eng_row, kor_sheet_row, eng_sheet_row, log_lines)

                            log_area.code('\n'.join(log_lines[-30:]))
                            progress_bar.progress(i / total)

                            if i < total:
                                time.sleep(4)

                        # 종료 보고서
                        end_time = datetime.now()
                        duration = str(end_time - start_time).split('.')[0]
                        status_text.empty()

                        st.success(
                            f'완료! 성공: {success} | 실패: {fail} | 스킵: {skip} | '
                            f'총: {total} | 소요: {duration}'
                        )
                        if fail_list:
                            st.warning(f'실패 목록: {", ".join(fail_list)}')

    with main_col_right:
        st.subheader('새로운 문제 추가')
        st.markdown("---")
        sheets_service_right = get_sheets_service()
        genai.configure(api_key=GEMINI_API_KEY)
        model_right = genai.GenerativeModel(MODEL_NAME)

        max_q_num = get_max_q_no(sheets_service_right)
        st.info(f"현재 마지막 문제 번호: Q.{max_q_num:03d}")
        st.caption("한글 또는 영문으로 입력하면 Pro_Kor(한글) + Pro_Eng(영문) 양쪽에 자동 동기화됩니다.")

        st.markdown("---")

        new_q_no = st.text_input('문제번호 (예: 112)', value=str(max_q_num + 1), key='new_q_no')
        new_q_text = st.text_area('문제 내용 (한글 또는 영문)', key='new_q_text')
        new_q_options = st.text_area('보기 (각 보기를 줄바꿈으로 구분)', key='new_q_options')

        if st.button('문제 추가 (양쪽 시트 동기화)', type='primary', use_container_width=True, key='add_new_question_button'):
            if not new_q_no.strip() or not new_q_text.strip() or not new_q_options.strip():
                st.error("문제번호, 문제 내용, 보기를 모두 입력해주세요.")
            else:
                try:
                    q_no_int = int(new_q_no.strip())

                    kor_header_btn, _ = read_sheet_data(sheets_service_right, SHEET_NAME_PRO_KOR)
                    eng_header_btn, _ = read_sheet_data(sheets_service_right, SHEET_NAME_PRO_ENG)

                    if not kor_header_btn or not eng_header_btn:
                        st.error("Pro_Kor 또는 Pro_Eng 시트 헤더를 읽을 수 없습니다.")
                    else:
                        add_new_question_synced(
                            model_right, sheets_service_right,
                            q_no_int, new_q_text.strip(), new_q_options.strip(),
                            kor_header_btn, eng_header_btn,
                        )
                        st.rerun()
                except ValueError:
                    st.error("문제번호는 유효한 숫자로 입력해주세요.")
                except Exception as e:
                    st.error(f"문제 추가 중 오류 발생: {e}")
