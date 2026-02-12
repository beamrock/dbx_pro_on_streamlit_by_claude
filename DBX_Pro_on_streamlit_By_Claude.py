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
SHEET_NAME = get_secret('SHEET_NAME', 'Pro_Kor')
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

SAFETY = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


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


def update_cell(sheets_service, row_num, subject, category, title):
    sheets_service.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=f"'{SHEET_NAME}'!A{row_num}:C{row_num}",
        valueInputOption='RAW',
        body={'values': [[subject, category, title]]}
    ).execute()


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


# --- Streamlit UI ---
st.set_page_config(page_title='DBX Pro 문제 분류', layout='centered')

# 사이드바 메뉴
MENU = {
    '문제 자동 분류': '🏷️',
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

# --- 페이지: 문제 자동 분류 ---
if selected_menu == '문제 자동 분류':
    st.title('Databricks Pro 시험 문제 자동 분류')

    col1, col2 = st.columns(2)
    with col1:
        start_question_number = st.number_input(
            '시작번호', min_value=1, max_value=999, value=1, step=1
        )
    with col2:
        end_question_number = st.number_input(
            '종료번호', min_value=1, max_value=999, value=111, step=1
        )

    st.caption('Overwrite (체크 시 기존 값이 있어도 덮어쓰기)')
    ow1, ow2, ow3 = st.columns(3)
    with ow1:
        ow_subject = st.checkbox('subject')
    with ow2:
        ow_category = st.checkbox('category')
    with ow3:
        ow_title = st.checkbox('title')

    if st.button('시작', type='primary', use_container_width=True):
        if start_question_number > end_question_number:
            st.error('시작번호가 종료번호보다 큽니다.')
        else:
            # 초기화
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(MODEL_NAME)
            sheets_service = get_sheets_service()

            # 시트 데이터 읽기
            result = sheets_service.spreadsheets().values().get(
                spreadsheetId=SPREADSHEET_ID,
                range=f"'{SHEET_NAME}'!A:H"
            ).execute()
            all_rows = result.get('values', [])
            header = all_rows[0]
            data_rows = all_rows[1:]

            COL = {h.strip().replace('\n', ''): i for i, h in enumerate(header)}
            IDX_QNUM    = COL.get('문제번호', 3)
            IDX_QTEXT   = COL.get('문제_KOR', 4)
            IDX_CHOICES = COL.get('보기_KOR', 5)
            IDX_REF     = COL.get('참고', 6)

            # 대상 행 필터링
            target_rows = []
            for idx, row in enumerate(data_rows):
                sheet_row = idx + 2
                q_num_str = row[IDX_QNUM] if len(row) > IDX_QNUM else ''
                try:
                    num = int(q_num_str.replace('Q.', '').strip())
                except Exception:
                    continue
                if start_question_number <= num <= end_question_number:
                    target_rows.append((sheet_row, num, row))

            total = len(target_rows)
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

                for i, (sheet_row, num, row) in enumerate(target_rows, 1):
                    q_label = f'Q.{num:03d}'

                    # 기존 값 확인 (A=subject, B=category, C=title)
                    existing_subj = row[0].strip() if len(row) > 0 else ''
                    existing_cat  = row[1].strip() if len(row) > 1 else ''
                    existing_ttl  = row[2].strip() if len(row) > 2 else ''

                    # 헤더행 제외
                    is_header = existing_subj == 'subject'

                    # 각 컬럼별로 업데이트가 필요한지 판단
                    need_subj = not existing_subj or is_header or ow_subject
                    need_cat  = not existing_cat  or is_header or ow_category
                    need_ttl  = not existing_ttl  or is_header or ow_title

                    if not need_subj and not need_cat and not need_ttl:
                        skip += 1
                        log_lines.append(f'[{i:03d}/{total}] {q_label} -> SKIP ({existing_subj})')
                        log_area.code('\n'.join(log_lines[-30:]))
                        progress_bar.progress(i / total)
                        continue

                    status_text.text(f'처리 중: {q_label} ({i}/{total})')

                    q_text    = row[IDX_QTEXT]   if len(row) > IDX_QTEXT   else ''
                    q_choices = row[IDX_CHOICES] if len(row) > IDX_CHOICES else ''
                    q_ref     = row[IDX_REF]     if len(row) > IDX_REF     else ''

                    subject, category, title, err_msg = classify_row(
                        model, q_label, q_text, q_choices, q_ref
                    )

                    if subject:
                        # 체크되지 않은 항목은 기존 값 유지
                        final_subj = subject          if need_subj else existing_subj
                        final_cat  = category or ''   if need_cat  else existing_cat
                        final_ttl  = title or ''      if need_ttl  else existing_ttl
                        update_cell(sheets_service, sheet_row, final_subj, final_cat, final_ttl)
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
