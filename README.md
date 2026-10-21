# ♻️ 回收小幫手

拍一張照片，告訴你它該丟哪裡、丟之前要做什麼。

這是一個 AI 回收物辨識的概念驗證。它用 Google Gemini 辨認照片中的物品，再依照一張分類規則表，告訴使用者該丟資源回收、特殊回收、一般垃圾還是廚餘，以及丟之前要倒空、沖洗或撕標籤。

## 功能

- **拍照或上傳照片辨識**：支援手機鏡頭直接拍照。
- **先猜猜看再看答案**：使用者先選自己會丟哪一桶，再對照 AI 的判斷，告訴他對不對、錯在哪裡。
- **依照片狀態提醒**：看到瓶子裡還有液體、便當盒有殘渣、寶特瓶沒撕標籤時，會額外提醒。
- **統計**：累計辨識件數、各類別比例、猜對率，以及最常分錯的品項。
- **示範模式**：沒有 API 金鑰時，可以用內建的範例照片試玩。

## 設計重點：AI 只負責「認東西」

分類規則不交給 AI 自由發揮。AI 只回答兩件事：

1. 這是清單中的哪一個品項（限定選項，不能自己編）。
2. 照片中看得到的狀態：有沒有液體、殘渣、標籤、破損。

至於該丟哪一桶、丟之前要做什麼，全部來自 [`recycle_helper/catalog.py`](recycle_helper/catalog.py) 這張規則表。這樣做的好處：

- **結果一致**：同一個品項永遠得到同樣的分類建議。
- **可以檢查**：規則寫在表裡，任何人都能逐條核對、修改。
- **AI 判斷錯時容易發現**：信心低於 60% 會提醒使用者再拍一次。

各縣市的回收規定可能不同，App 也會提醒使用者以當地環保局公告為準。

## 本機執行

```sh
pip install -r requirements.txt
streamlit run app.py
```

沒有設定金鑰時會自動進入示範模式。要使用 AI 辨識，在 `.streamlit/secrets.toml` 加入：

```toml
GEMINI_API_KEY = "你的金鑰"
# 選填，預設是 gemini-3.5-flash-lite
# GEMINI_MODEL = "gemini-3.5-flash-lite"
```

金鑰可以在 [Google AI Studio](https://aistudio.google.com/) 申請。`secrets.toml` 已經列在 `.gitignore`，不會被上傳。

## 部署到 Streamlit Community Cloud

1. 把這個專案推到 GitHub。
2. 到 [Streamlit Community Cloud](https://share.streamlit.io/) 建立新 App，選這個 repo，主程式填 `app.py`。
3. 在 App 的 **Settings > Secrets** 貼上 `GEMINI_API_KEY = "你的金鑰"`。

## 測試

```sh
pip install pytest
python -m pytest
```

## 專案結構

```text
app.py                     Streamlit 畫面
recycle_helper/
  catalog.py               分類規則表
  classifier.py            Gemini 辨識與示範模式
  result.py                解讀 AI 回覆，產生處理建議
  stats.py                 統計
samples/                   示範模式用的範例圖（插畫，不是實物照片）
tests/                     自動測試
```
