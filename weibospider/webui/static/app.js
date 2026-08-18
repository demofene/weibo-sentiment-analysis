const state = {
  currentJobId: null,
  pollTimer: null,
  files: null,
};

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function splitInput(text) {
  return text
    .split(/\r?\n|,/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function extractWeiboIds(rawInput) {
  const lines = splitInput(rawInput);
  const ids = [];
  const patterns = [
    // PC版：https://weibo.com/1644114654/OM4rNu3S3
    /weibo\.com\/\d+\/([A-Za-z0-9]+)/i,
    // 移动版：https://m.weibo.cn/detail/4637843577309754
    /m\.weibo\.cn\/detail\/([A-Za-z0-9]+)/i,
    // 带 mid 参数：https://weibo.com/aj/mblog?mid=4637843577309754
    /[?&]mid=(\d+)/i,
    // 新浪短链：https://t.cn/A6xxxxxx （无法直接提取，保留整体，或跳过）
    // 纯 ID（字母数字组合长度>5）
  ];

  for (const line of lines) {
    let found = false;
    for (const pattern of patterns) {
      const match = line.match(pattern);
      if (match) {
        ids.push(match[1]);
        found = true;
        break;
      }
    }
    if (!found) {
      // 当做纯 ID（如 OM4rNu3S3 或 4637843577309754）
      ids.push(line);
    }
  }
  return ids;
}

function setStatus(message, kind = "idle") {
  const chip = document.getElementById("jobStatusChip");
  chip.textContent = message;
  chip.className = `job-chip ${kind}`;
}

function setJobMeta(message) {
  document.getElementById("jobMeta").textContent = message;
}

function setLogs(lines) {
  const logView = document.getElementById("jobLogs");
  logView.textContent = Array.isArray(lines) && lines.length ? lines.join("\n") : "等待任务启动…";
  logView.scrollTop = logView.scrollHeight;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
    },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with status ${response.status}`);
  }
  return payload;
}

function optionMarkup(file) {
  return `<option value="${escapeHtml(file.absolute_path)}">${escapeHtml(file.relative_path)}</option>`;
}

function renderFileList(targetId, files) {
  const target = document.getElementById(targetId);
  if (!files.length) {
    target.innerHTML = "<li>还没有文件</li>";
    return;
  }

  target.innerHTML = files
    .map(
      (file) => `
        <li>
          <span class="file-path">${escapeHtml(file.relative_path)}</span>
          <span class="file-meta">${escapeHtml(file.updated_at)} · ${Math.round(file.size_bytes / 1024)} KB</span>
        </li>
      `,
    )
    .join("");
}

function populateSelect(selectId, files, emptyLabel) {
  const select = document.getElementById(selectId);
  if (!files.length) {
    select.innerHTML = `<option value="">${escapeHtml(emptyLabel)}</option>`;
    return;
  }
  select.innerHTML = files.map(optionMarkup).join("");
}

async function refreshFiles() {
  state.files = await fetchJson("/api/files");

  document.getElementById("rawCount").textContent = state.files.raw.length;
  document.getElementById("cleanedCount").textContent = state.files.cleaned.length;
  document.getElementById("resultCount").textContent = state.files.results.length;

  populateSelect(
    "rawFileSelect",
    state.files.raw.filter((file) => file.name.endsWith(".jsonl")),
    "暂无可用文件",
  );
  populateSelect(
    "cleanedFileSelect",
    state.files.cleaned.filter((file) => file.name.endsWith(".json")),
    "暂无可用文件",
  );

  renderFileList("rawFilesList", state.files.raw);
  renderFileList("cleanedFilesList", state.files.cleaned);
  renderFileList("resultFilesList", state.files.results);

  const defaultModel = state.files.defaults?.model_dir?.absolute_path || "";
  if (!document.getElementById("fullModelDir").value) {
    document.getElementById("fullModelDir").value = defaultModel;
  }
  if (!document.getElementById("analyzeModelDir").value) {
    document.getElementById("analyzeModelDir").value = defaultModel;
  }
  if (!document.getElementById("keywordModelDir").value) {
    document.getElementById("keywordModelDir").value = defaultModel;
  }
}

function renderAnalysisSummary(summary, topExamples, wordcloud) {
  const container = document.getElementById("analysisSummary");
  if (!summary) {
    container.innerHTML = "暂无分析结果，运行一次分析吧。";
    return;
  }

  const labels = ["positive", "neutral", "negative"];
  const displayNames = {
    positive: "正向",
    neutral: "中性",
    negative: "负向",
  };

  // ---------- 左半部分：总体判断 + 指标 ----------
  const tendencyHtml = `
    <div class="summary-card">
      <h3>总体判断</h3>
      <div class="tendency-value">${escapeHtml(summary.overall_tendency.label)}</div>
      <div class="metric-list">
        <div class="metric-row"><span>主导占比</span><strong>${(summary.overall_tendency.dominant_ratio * 100).toFixed(1)}%</strong></div>
        <div class="metric-row"><span>平均置信度</span><strong>${(summary.average_confidence * 100).toFixed(1)}%</strong></div>
        <div class="metric-row"><span>平均类别边距</span><strong>${summary.average_margin.toFixed(3)}</strong></div>
        <div class="metric-row"><span>情绪指数</span><strong>${summary.sentiment_index.toFixed(3)}</strong></div>
        <div class="metric-row"><span>评论/微博总数</span><strong>${summary.total_comments}</strong></div>
      </div>
    </div>
  `;

  // ---------- 右半部分：环形图（用 canvas 占位） ----------
  const chartHtml = `
    <div class="summary-card">
      <h3>情感分布</h3>
      <div class="chart-container" style="max-width:260px; margin:0 auto;">
        <canvas id="sentimentChart"></canvas>
      </div>
    </div>
  `;

  // ---------- 总体词云卡片 ----------
  const wordcloudHtml = `
    <div class="summary-card summary-card-full">
      <h3>高频词云</h3>
      <div class="wordcloud-container">
        <canvas id="wordcloudCanvas" width="600" height="300"></canvas>
      </div>
    </div>
  `;
  // ---------- 示例微博（三列卡片，保持不变） ----------
  const examples = labels
    .map((label) => {
      const title = displayNames[label];
      const items = (topExamples?.[label] || [])
        .map(
          (item) => `
            <div class="example-item">
              <p class="example-text">${escapeHtml(item.text)}</p>
              <div class="example-meta">置信度 ${(item.confidence * 100).toFixed(1)}% · 点赞 ${item.like_counts}</div>
            </div>
          `,
        )
        .join("");
      return `
        <div class="example-column">
          <h3>${escapeHtml(title)}</h3>
          <div class="example-list">${items || "<div class='example-item'>暂无样本数据</div>"}</div>
        </div>
      `;
    })
    .join("");

  container.innerHTML = `
    <div class="summary-grid">
      ${tendencyHtml}
      ${chartHtml}
    </div>
    ${wordcloudHtml} 
    <div class="examples-grid">${examples}</div>
  `;

  // ---------- 绘制环形图 ----------
  const ctx = document.getElementById("sentimentChart");
  if (ctx) {
    // 销毁旧图表（如果重复渲染时会自动处理）
    const existing = Chart.getChart(ctx);
    if (existing) existing.destroy();

    const dataValues = labels.map((l) => summary.distribution[l].count);
    new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: labels.map((l) => displayNames[l]),
        datasets: [
          {
            data: dataValues,
            backgroundColor: ["#10b981", "#64748b", "#ef4444"],
            borderColor: "#fff",
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              padding: 16,
              usePointStyle: true,
              pointStyleWidth: 10,
              font: { size: 13 },
            },
          },
          tooltip: {
            callbacks: {
              label: (ctx) =>
                `${ctx.label}: ${ctx.raw} 条 (${((ctx.raw / summary.total_comments) * 100).toFixed(1)}%)`,
            },
          },
        },
      },
    });
  }
  // ---------- 绘制总体词云 ----------
  const canvas = document.getElementById("wordcloudCanvas");
  if (canvas) {
    if (wordcloud && wordcloud.length) {
      canvas.style.display = '';   // 确保可见
      const list = wordcloud.map(([word, weight]) => [word, weight]);
      WordCloud(canvas, {
        list: list,
        gridSize: 8,
        weightFactor: 1.2,
        fontFamily: "PingFang SC, Microsoft YaHei, sans-serif",
        color: () => `hsl(${Math.floor(Math.random() * 360)}, 60%, 50%)`,
        backgroundColor: "transparent",
        rotateRatio: 0.3,
        shape: "circle",
        minSize: 6,
        clearCanvas: true,
      });
      // 如果有之前的“未生成”提示，移除
      const tip = canvas.parentNode?.querySelector('.no-data-tip');
      if (tip) tip.remove();
    } else {
      canvas.style.display = 'none';
      // 插入提示（加 class 便于后续删除）
      if (!canvas.parentNode?.querySelector('.no-data-tip')) {
        const tip = document.createElement("p");
        tip.className = 'no-data-tip';
        tip.textContent = "暂无词云数据";
        canvas.parentNode.appendChild(tip);
      }
    }
  }
}

function extractSummaryFromJob(job) {
  if (!job?.result) return null;
  if (job.kind === "analyze" || job.kind === "keyword-sentiment") {
    return {
      summary: job.result.summary,
      topExamples: job.result.top_examples,
      wordcloud: job.result.wordcloud || {},
    };
  }
  if (job.kind === "full-comment") {
    return {
      summary: job.result.analysis?.summary,
      topExamples: job.result.analysis?.top_examples,
      wordcloud: job.result.analysis?.wordcloud || {},
    };
  }
  if (job.kind === "full-keyword-sentiment") {
    return {
      summary: job.result.analysis?.summary,
      topExamples: job.result.analysis?.top_examples,
      wordcloud: job.result.analysis?.wordcloud || {},
    };
  }
  return null;
}

async function pollJob(jobId) {
  const job = await fetchJson(`/api/jobs/${jobId}`);
  state.currentJobId = jobId;

  setStatus(
    job.status === "queued" ? "排队中" :
    job.status === "running" ? "运行中" :
    job.status === "succeeded" ? "已完成" : "失败",
    job.status,
  );
  setJobMeta(`${job.kind} · ${job.updated_at}`);
  setLogs(job.logs);

  const summaryPayload = extractSummaryFromJob(job);
  if (summaryPayload?.summary) {
    renderAnalysisSummary(summaryPayload.summary, summaryPayload.topExamples, summaryPayload.wordcloud);
  }

  if (job.status === "succeeded" || job.status === "failed") {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    await refreshFiles();
  }
}

async function startJob(endpoint, payload) {
  const job = await fetchJson(endpoint, {
    method: "POST",
    body: JSON.stringify(payload),
  });

  clearInterval(state.pollTimer);
  setStatus("已提交", "running");
  setJobMeta(`${job.kind} · ${job.created_at}`);
  setLogs(["任务已提交，排队等待中…"]);
  state.currentJobId = job.id;
  state.pollTimer = setInterval(() => {
    pollJob(job.id).catch((error) => {
      setStatus("失败", "failed");
      setJobMeta("获取任务状态失败");
      setLogs([String(error)]);
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    });
  }, 1500);
  await pollJob(job.id);
}

function bindModeToggle() {
  const modeSelect = document.getElementById("crawlMode");
  const commentFields = document.getElementById("commentFields");
  const userFields = document.getElementById("userFields");
  const keywordFields = document.getElementById("keywordFields");

  const refresh = () => {
    commentFields.classList.toggle("hidden", modeSelect.value !== "comment");
    userFields.classList.toggle("hidden", modeSelect.value !== "user");
    keywordFields.classList.toggle("hidden", modeSelect.value !== "tweet_by_keyword");
  };

  modeSelect.addEventListener("change", refresh);
  refresh();
}

function bindForms() {
  document.getElementById("fullCommentForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const modelName = document.getElementById("fullModelSelect").value;  // 新增
    await startJob("/api/jobs/full-comment", {
      tweet_ids: extractWeiboIds(document.getElementById("fullTweetIds").value),
      count: Number(document.getElementById("fullCount").value || 20),
      batch_size: Number(document.getElementById("fullBatchSize").value || 32),
      model_dir: document.getElementById("fullModelDir").value.trim(),
      model_name: modelName,   // 新增
    });
  });

  document.getElementById("crawlForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const mode = document.getElementById("crawlMode").value;
    const payload = { mode };

    if (mode === "comment") {
      payload.tweet_ids = extractWeiboIds(document.getElementById("crawlTweetIds").value);
      payload.count = Number(document.getElementById("crawlCount").value || 20);
    } else if (mode === "user") {
      payload.user_ids = splitInput(document.getElementById("crawlUserIds").value);
    } else {
      payload.keywords = splitInput(document.getElementById("crawlKeywords").value);
      payload.start_time = document.getElementById("crawlStartTime").value.trim();
      payload.end_time = document.getElementById("crawlEndTime").value.trim();
      payload.split_by_hour = document.getElementById("crawlSplitByHour").checked;
    }

    await startJob("/api/jobs/crawl", payload);
  });

  document.getElementById("cleanForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    await startJob("/api/jobs/clean", {
      raw_file: document.getElementById("rawFileSelect").value,
    });
  });

  document.getElementById("analyzeForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const modelName = document.getElementById("analyzeModelSelect").value;  // 新增
    await startJob("/api/jobs/analyze", {
      cleaned_file: document.getElementById("cleanedFileSelect").value,
      model_dir: document.getElementById("analyzeModelDir").value.trim(),
      model_name: modelName,   // 使用变量
      batch_size: Number(document.getElementById("analyzeBatchSize").value || 32),
      top_examples_per_label: Number(document.getElementById("topExamplesPerLabel").value || 5),
    });
  });

    // 关键词一键分析
  document.getElementById("fullKeywordForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const keyword = document.getElementById("keywordInput").value.trim();
    if (!keyword) { alert("请输入关键词"); return; }
    const modelName = document.getElementById("keywordModelSelect").value;  // 新增
    await startJob("/api/jobs/full-keyword-sentiment", {
      keyword: keyword,
      start_time: document.getElementById("keywordStartTime").value.trim(),
      end_time: document.getElementById("keywordEndTime").value.trim(),
      batch_size: Number(document.getElementById("keywordBatchSize").value || 32),
      top_examples_per_label: Number(document.getElementById("keywordTopExamples").value || 5),
      split_by_hour: document.getElementById("keywordSplitByHour").checked,
      model_dir: document.getElementById("keywordModelDir").value.trim(),
      model_name: modelName,   // 新增
    });
  });
}

async function bootstrap() {
  bindModeToggle();
  bindForms();
  await refreshFiles();
}

bootstrap().catch((error) => {
  setStatus("失败", "failed");
  setJobMeta("页面初始化失败，请刷新后重试");
  setLogs([String(error)]);
});
