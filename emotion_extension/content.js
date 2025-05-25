const API_KEY = "";  // ⬅️ 替換為你的 YouTube API 金鑰

// 從目前網址中擷取影片 ID
function extractVideoIdFromUrl() {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get("v");
}

// 使用 YouTube Data API v3 抓取所有留言
async function fetchComments(videoId) {
  let comments = [];
  let nextPageToken = "";

  while (true) {
    const apiUrl = `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${nextPageToken}&key=${API_KEY}`;

    try {
      const res = await fetch(apiUrl);
      const data = await res.json();
      if (!data.items) break;

      data.items.forEach(item => {
        const text = item.snippet.topLevelComment.snippet.textDisplay;
        comments.push(text);
      });

      if (!data.nextPageToken) break;
      nextPageToken = data.nextPageToken;
    } catch (err) {
      console.error("抓留言失敗：", err);
      break;
    }
  }

  return comments;
}

function insertResultPanel(result, isPartial = false) {
  if (document.getElementById("emotion-panel")) {
    document.getElementById("emotion-panel").remove();
  }

  const panel = document.createElement("div");
  panel.id = "emotion-panel";
  panel.style.position = "fixed";
  panel.style.top = "100px";
  panel.style.right = "20px";
  panel.style.width = "200px";
  panel.style.backgroundColor = "#fff";
  panel.style.border = "2px solid #000";
  panel.style.padding = "10px";
  panel.style.boxShadow = "0 0 10px rgba(0,0,0,0.5)";
  panel.style.zIndex = 9999;
  panel.style.fontFamily = "Arial, sans-serif";

  const title = document.createElement("h4");
  title.textContent = isPartial ? "部分情緒分析結果" : "完整情緒分析結果";
  title.style.marginTop = "0";
  panel.appendChild(title);

  for (const [emotion, value] of Object.entries(result)) {
    const p = document.createElement("p");
    p.textContent = `${emotion}: ${(value * 100).toFixed(1)}%`;
    panel.appendChild(p);
  }

  document.body.appendChild(panel);
}

async function fetchAndAnalyze() {
  const videoId = extractVideoIdFromUrl();
  if (!videoId) {
    console.warn("無法從網址中擷取影片 ID");
    return;
  }

  console.log("正在從 YouTube API 抓留言...");
  const comments = await fetchComments(videoId);

  console.log(`留言，共 ${comments.length} 則`);

  fetch("http://localhost:8000/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ comments: comments })
  })
    .then(response => response.json())
    .then(data => {
      console.log("部分分析結果：", data.result);
      insertResultPanel(data.result, data.partial);

      if (data.partial) {
        pollFullResult();
      }
    })
    .catch(err => {
      console.error("傳送失敗:", err);
    });
}

function pollFullResult() {
  const intervalId = setInterval(() => {
    fetch("http://localhost:8000/full_result")
      .then(response => response.json())
      .then(data => {
        if (data.partial === false) {
          console.log("完整分析結果：", data.result);
          insertResultPanel(data.result, false);
          clearInterval(intervalId);
        } else {
          console.log("完整分析結果尚未準備好");
        }
      })
      .catch(err => {
        console.error("輪詢完整結果失敗:", err);
        clearInterval(intervalId);
      });
  }, 5000);
}

fetchAndAnalyze();
