import glob
import os
import subprocess
import time

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors

# ============ 設定區（想改的東西都在這裡）============
AUDIO_FILE = "test.m4a"      # 要處理的音檔（mp3、m4a、wav 都可以，記得改成你的檔名）
CHUNK_MINUTES = 30           # 每幾分鐘切一段
MEETING_TITLE = ""           # 會議標題，例如 "0911組長會議"；留空的話由 AI 依內容擬定
# ====================================================

# 1. 讀取 .env 裡的金鑰
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("找不到金鑰！請檢查 .env 檔案中的 GEMINI_API_KEY")

# 2. 建立 Gemini 連線
client = genai.Client(api_key=api_key)
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def with_retry(action, *args, waits=(15, 30, 60, 90, 120), **kwargs):
    """執行一個需要連網的動作。

    遇到下面兩種情況，會等一下自動重試（最多 5 次）：
      1. Google 伺服器太忙（503）或你送太快（429、500）
      2. 你的網路連線中途斷掉
    其他錯誤（例如金鑰錯誤、模型名稱不對）重試也沒用，會直接停下來顯示錯誤。
    """
    for attempt in range(len(waits) + 1):
        try:
            return action(*args, **kwargs)
        except (errors.APIError, httpx.TransportError) as e:
            is_api_error = isinstance(e, errors.APIError)
            if is_api_error and e.code not in (429, 500, 503):
                raise
            if attempt == len(waits):
                raise
            reason = f"Google 伺服器忙碌中（{e.code}）" if is_api_error else "網路連線中斷"
            print(f"  {reason}，{waits[attempt]} 秒後自動重試（{attempt + 1}/{len(waits)}）...")
            time.sleep(waits[attempt])


# 主要模型一直排不到（太忙）或名稱不能用時，會依序改用這些備用模型
# （gemini-2.5-flash 已不開放新使用者，所以不放進來）
FALLBACK_MODELS = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]


def ask_gemini(contents):
    """問 Gemini 一個問題，回傳它的文字回答。主要模型不行就自動換備用模型。"""
    models = [MODEL] + [m for m in FALLBACK_MODELS if m != MODEL]
    last_error = None
    for i, model in enumerate(models):
        try:
            response = with_retry(
                client.models.generate_content,
                waits=(10, 20, 40),   # 每個模型最多試 4 次，約 1 分鐘
                model=model,
                contents=contents,
            )
            if i > 0:
                print(f"  （這次改用備用模型：{model}）")
            return response.text
        except errors.APIError as e:
            if e.code not in (404, 429, 500, 503):
                raise  # 例如金鑰錯誤，換模型也沒用
            last_error = e
            print(f"  模型 {model} 目前無法使用（{e.code}），換下一個模型...")
    raise last_error


def split_audio(path):
    """用 ffmpeg 把音檔切成每 30 分鐘一段，回傳切好的檔名清單。"""
    # 先清掉上次留下的暫存檔
    for old in glob.glob("temp_chunk_*.mp3"):
        os.remove(old)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", path,
                "-f", "segment",
                "-segment_time", str(CHUNK_MINUTES * 60),
                "-vn", "-ac", "1", "-ar", "16000",   # 只留聲音、單聲道，不管原本是什麼格式都轉成 mp3
                "temp_chunk_%03d.mp3",
            ],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        raise SystemExit("找不到 ffmpeg！請在終端機輸入：winget install ffmpeg ，裝完後把 VSCode 關掉重開。")

    return sorted(glob.glob("temp_chunk_*.mp3"))


def transcribe_chunk(chunk_path):
    """把一小段音檔交給 Gemini，回傳逐字稿文字。"""
    # 上傳音檔到 Google
    uploaded = with_retry(client.files.upload, file=chunk_path)

    # 等 Google 處理完檔案（通常幾秒）
    while uploaded.state is not None and uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = with_retry(client.files.get, name=uploaded.name)

    try:
        return ask_gemini([
            uploaded,
            "請將這段錄音轉換為精確的繁體中文逐字稿，只要輸出逐字稿內容就好，不要加上任何額外的說明。",
        ])
    finally:
        # 不管成功或失敗，都把 Google 上的暫存檔刪掉（刪不掉也沒關係，Google 會自動清除）
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            print("  （Google 上的暫存檔沒刪成功，可以忽略）")


def make_transcript(audio_path):
    """步驟一：音檔 → 逐字稿。只要有任何一段失敗就停下來，不會存出殘缺的逐字稿。"""
    print("正在切割音檔...")
    chunks = split_audio(audio_path)
    if not chunks:
        raise SystemExit("音檔切割後沒有產生任何內容，請確認錄音檔是不是壞掉或是空的。")
    print(f"共切成 {len(chunks)} 段")

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"正在處理第 {i}/{len(chunks)} 段...")
        try:
            text = (transcribe_chunk(chunk) or "").strip()
        except Exception as e:
            raise SystemExit(
                f"\n第 {i} 段轉逐字稿失敗：{e}\n"
                "為了避免存出不完整的逐字稿，程式先停下來。稍後重新執行就會重新處理。"
            )
        if not text:
            raise SystemExit(f"\n第 {i} 段 Gemini 沒有回傳任何文字（可能是這段沒有人聲，或被安全機制擋掉）。")
        parts.append(text)
        os.remove(chunk)  # 刪掉本機暫存檔
    return "\n".join(parts)


MINUTES_PROMPT = """
你是一位專業的會議記錄助理。請根據下方的會議逐字稿，整理出一份「詳細、有層次」的會議記錄，使用繁體中文（台灣用語）。

【輸出格式】使用 Markdown 巢狀條列，每一層縮排 4 個空格，結構如下：

# {title_rule}

## 當日會議討論內容
- 主題或單位名稱
    - 該主題的重點事項
        - 補充細節、原因或條件
- 下一個主題或單位名稱
    - ……

## 待辦事項
（格式與規則見下方）

【當日會議討論內容 的寫法】
1. 依「主題」或「單位／組別」分組，順序照會議討論的先後；分組名稱沿用與會者實際使用的稱呼。
2. 每個分組下用 2～3 層條列。第二層寫結論或重點；需要說明原因、背景、條件、做法時，放到第三層。
3. 內容要具體：保留數字、日期、期限、專案名稱、系統名稱、人名，以及「為什麼這樣決定」的原因，不要只寫「討論了某某事」。
4. 主管的指示、規範或需要全體注意的事項，在該條開頭標示「（重要指示）」。
5. 使用書面語，語氣精簡正式，不要口語化，不要寫「他說」「然後」這類敘述；閒聊和與工作無關的內容略過。
6. 只有一句話能說完的主題，就只寫一兩條，不要為了湊層次而拆得很碎。
7. 只根據逐字稿內容，不可編造沒有出現的事實、數字、日期或人名。專有名詞不確定時保留原樣，並在後面加上（？）。

【待辦事項】
每一項都要寫出：負責人、具體任務、期限；逐字稿沒提到就寫「未提及」，不要自己編造。

【格式示範】以下只是版面示範，內容與本次會議無關，不可寫進輸出：
- 某某專案
    - 目前進度約六成，預計 10 月底前完成第一階段驗證。
        - 因測試資料不足，需再補收 50 筆。
    - （重要指示）對外文件一律使用正式書面語。

【會議逐字稿】：
{transcript}
"""


def make_minutes(transcript):
    """步驟二：逐字稿 → 會議記錄 + 待辦事項"""
    print("正在整理會議記錄與待辦事項...")
    if MEETING_TITLE.strip():
        title_rule = MEETING_TITLE.strip()
    else:
        title_rule = "會議標題（依內容擬定，簡短，例如「專案進度會議」）"
    prompt = MINUTES_PROMPT.format(title_rule=title_rule, transcript=transcript)
    return ask_gemini(prompt)


if __name__ == "__main__":
    if not os.path.exists(AUDIO_FILE):
        raise SystemExit(f"找不到 {AUDIO_FILE}！請把錄音檔放進這個資料夾。")

    os.makedirs("output", exist_ok=True)

    transcript = ""
    if os.path.exists("output/逐字稿.txt"):
        with open("output/逐字稿.txt", encoding="utf-8") as f:
            transcript = f.read()
        if transcript.strip():
            print("發現已經做好的 output/逐字稿.txt，直接使用（想重做的話，把這個檔案刪掉再執行）")
        else:
            print("發現 output/逐字稿.txt 是空的（可能是上次失敗留下的），重新產生...")

    if not transcript.strip():
        transcript = make_transcript(AUDIO_FILE)
        with open("output/逐字稿.txt", "w", encoding="utf-8") as f:
            f.write(transcript)

    minutes = make_minutes(transcript)
    with open("output/會議記錄.md", "w", encoding="utf-8") as f:
        f.write(minutes)

    print("\n完成！結果存在 output 資料夾：")
    print("  output/逐字稿.txt")
    print("  output/會議記錄.md")