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

// 抓完留言後送出給後端分析
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
      console.log("分析結果：", data);
    })
    .catch(err => {
      console.error("傳送失敗:", err);
    });
}

fetchAndAnalyze();
