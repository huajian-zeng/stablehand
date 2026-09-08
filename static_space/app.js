(() => {
  "use strict";
  const cases = window.STABLEHAND_CASES || [];
  const $ = (id) => document.getElementById(id);
  const video = $("comparison-video");
  const timeline = $("timeline");
  const grid = $("case-grid");
  const message = $("video-message");
  let activeFilter = "all";
  let activeCase = cases[0];
  let frameCallback = null;
  let selectionVersion = 0;
  const visibleCases = () =>
    cases.filter(
      (item) => activeFilter === "all" || item.dataset === activeFilter,
    );
  const pad = (number, size = 2) => String(number).padStart(size, "0");
  const formatTime = (time) =>
    `${pad(Math.floor(time / 60))}:${pad(Math.floor(time % 60))}`;

  if (!activeCase) {
    message.textContent =
      "The example collection is unavailable. Please reload the page.";
    message.hidden = false;
    return;
  }

  function updatePosition() {
    const collection = visibleCases();
    const position = collection.findIndex((item) => item.id === activeCase.id);
    $("sequence-position").innerHTML =
      `${pad(position + 1)} <span>/ ${pad(collection.length)}</span>`;
  }

  function renderCards() {
    grid.replaceChildren();
    visibleCases().forEach((item) => {
      const selected = item.id === activeCase.id;
      const card = document.createElement("button");
      card.type = "button";
      card.className = `case-card${selected ? " selected" : ""}`;
      card.dataset.id = item.id;
      card.setAttribute(
        "aria-label",
        `Play ${item.dataset} ${item.id}, ${item.duration} seconds`,
      );
      card.setAttribute("aria-pressed", String(selected));
      card.innerHTML = `<div class="thumbnail-wrap"><img src="${item.thumbnail}" alt="" loading="lazy" width="480" height="435"><span class="thumb-shade"></span><span class="thumb-dataset">${item.dataset}</span><span class="thumb-duration">${formatTime(item.duration)}</span><span class="thumbnail-play"><svg class="icon"><use href="#i-play"/></svg></span></div><div class="card-caption"><span>Sequence ${item.id.replace("clip-", "")}</span>${selected ? '<span class="selected-indicator"><i></i>Selected</span>' : `<small>${item.frames} frames</small>`}</div>`;
      card.addEventListener("click", () => {
        selectCase(item, true);
        if ($("player").getBoundingClientRect().bottom < 80) {
          $("player").scrollIntoView({
            behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
              ? "instant"
              : "smooth",
            block: "start",
          });
        }
      });
      grid.append(card);
    });
    updatePosition();
  }

  function syncTimeline() {
    const duration = Number.isFinite(video.duration)
      ? video.duration
      : activeCase.duration;
    const time = Math.min(video.currentTime || 0, duration);
    timeline.max = duration;
    timeline.step = 1 / activeCase.fps;
    timeline.value = time;
    timeline.style.setProperty(
      "--progress",
      `${duration ? (time / duration) * 100 : 0}%`,
    );
    timeline.setAttribute(
      "aria-valuetext",
      `${time.toFixed(2)} of ${duration.toFixed(2)} seconds`,
    );
    $("current-time").textContent = formatTime(time);
    $("total-time").textContent = formatTime(duration);
    const frame = Math.min(
      activeCase.frames,
      Math.floor(time * activeCase.fps + 0.001) + 1,
    );
    $("frame-display").textContent =
      `FRAME ${pad(frame, 3)} / ${activeCase.frames}`;
  }

  function scheduleFrame() {
    if (video.requestVideoFrameCallback && frameCallback === null) {
      frameCallback = video.requestVideoFrameCallback(() => {
        frameCallback = null;
        syncTimeline();
        if (!video.paused) scheduleFrame();
      });
    }
  }

  function updatePlayButton() {
    const playing = !video.paused && !video.ended;
    $("play-toggle").setAttribute(
      "aria-label",
      playing ? "Pause video" : "Play video",
    );
    $("play-toggle")
      .querySelector("use")
      .setAttribute("href", playing ? "#i-pause" : "#i-play");
    if (playing) scheduleFrame();
  }

  async function play() {
    const version = selectionVersion;
    try {
      await video.play();
      if (version === selectionVersion) message.hidden = true;
    } catch (error) {
      if (version !== selectionVersion || error.name === "AbortError") return;
      message.textContent =
        "Playback could not start. Press play to try again.";
      message.hidden = false;
    }
  }

  function togglePlay() {
    if (video.paused || video.ended) {
      if (video.ended) video.currentTime = 0;
      play();
    } else video.pause();
  }

  function selectCase(item, autoplay = false) {
    const focusId = document.activeElement?.closest(".case-card")?.dataset.id;
    selectionVersion += 1;
    video.pause();
    activeCase = item;
    message.hidden = true;
    video.poster = item.poster;
    video.src = item.video;
    video.load();
    video.playbackRate = Number($("playback-speed").value);
    $("player-dataset").textContent = item.dataset;
    $("player-sequence").textContent =
      `SEQUENCE ${item.id.replace("clip-", "")}`;
    $("sequence-title").textContent =
      item.dataset === "HOT3D"
        ? "Everyday interactions"
        : "Hands in articulated motion";
    $("sequence-description").innerHTML =
      `${item.dataset} · ${item.id} <span> / </span> ${item.duration} seconds <span> / </span> ${item.frames} frames · ${item.fps} fps`;
    $("selection-status").textContent =
      `Selected ${item.dataset} ${item.id}. ${item.duration} seconds, ${item.frames} frames.`;
    syncTimeline();
    renderCards();
    if (focusId)
      grid
        .querySelector(`[data-id="${focusId}"]`)
        ?.focus({ preventScroll: true });
    // Preserve dataset and sequence in the URL without requiring a server router.
    try {
      history.replaceState(
        null,
        "",
        `#${item.dataset.toLowerCase()}/${item.id}`,
      );
    } catch (_) {
      /* file URL previews can disallow history updates */
    }
    if (autoplay) play();
  }

  function navigate(direction) {
    const collection = visibleCases();
    const index = collection.findIndex((item) => item.id === activeCase.id);
    selectCase(
      collection[(index + direction + collection.length) % collection.length],
      true,
    );
  }

  function stepFrame(direction) {
    if (video.readyState < 1) return;
    video.pause();
    const frame =
      Math.floor(video.currentTime * activeCase.fps + 0.001) + direction;
    video.currentTime =
      Math.max(0, Math.min(activeCase.frames - 1, frame)) / activeCase.fps;
    syncTimeline();
  }

  document.querySelectorAll("[data-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      activeFilter = button.dataset.filter;
      document.querySelectorAll("[data-filter]").forEach((filter) => {
        const selected = filter === button;
        filter.classList.toggle("active", selected);
        filter.setAttribute("aria-pressed", String(selected));
      });
      if (!visibleCases().some((item) => item.id === activeCase.id))
        selectCase(visibleCases()[0]);
      else renderCards();
    });
  });
  $("play-toggle").addEventListener("click", togglePlay);
  $("video-stage").addEventListener("click", togglePlay);
  $("previous-example").addEventListener("click", () => navigate(-1));
  $("next-example").addEventListener("click", () => navigate(1));
  $("frame-prev").addEventListener("click", () => stepFrame(-1));
  $("frame-next").addEventListener("click", () => stepFrame(1));
  timeline.addEventListener("input", () => {
    if (video.readyState >= 1) video.currentTime = Number(timeline.value);
    syncTimeline();
  });
  $("playback-speed").addEventListener("change", (event) => {
    video.playbackRate = Number(event.target.value);
  });
  $("loop-toggle").addEventListener("click", () => {
    video.loop = !video.loop;
    $("loop-toggle").classList.toggle("is-active", video.loop);
    $("loop-toggle").setAttribute("aria-pressed", String(video.loop));
  });
  $("fullscreen-toggle").addEventListener("click", async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if ($("player").requestFullscreen)
        await $("player").requestFullscreen();
      else if (video.webkitEnterFullscreen) video.webkitEnterFullscreen();
      else throw new Error("Fullscreen unavailable");
    } catch (_) {
      message.textContent =
        "Fullscreen is unavailable in this preview. Open the gallery in a browser tab.";
      message.hidden = false;
    }
  });
  ["loadedmetadata", "timeupdate", "seeked", "durationchange"].forEach(
    (event) => video.addEventListener(event, syncTimeline),
  );
  ["play", "pause", "ended"].forEach((event) =>
    video.addEventListener(event, updatePlayButton),
  );
  video.addEventListener("error", () => {
    message.textContent =
      "This video could not be loaded. Select another example or reload the page.";
    message.hidden = false;
  });
  document.addEventListener("keydown", (event) => {
    if (
      event.target.closest(
        'button,a,input,select,textarea,[contenteditable="true"]',
      ) ||
      event.ctrlKey ||
      event.metaKey ||
      event.altKey
    )
      return;
    if (event.code === "Space") {
      event.preventDefault();
      togglePlay();
    }
    if (event.code === "ArrowLeft" || event.code === "ArrowRight") {
      event.preventDefault();
      stepFrame(event.code === "ArrowLeft" ? -1 : 1);
    }
    if (event.code === "Home") {
      event.preventDefault();
      video.pause();
      video.currentTime = 0;
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) video.pause();
  });
  window.addEventListener("hashchange", () => {
    const linked = cases.find(
      (item) => `#${item.dataset.toLowerCase()}/${item.id}` === location.hash,
    );
    if (!linked || linked.id === activeCase.id) return;
    // A shared link may point outside the currently filtered collection.
    if (activeFilter !== "all" && activeFilter !== linked.dataset) {
      activeFilter = "all";
      document.querySelectorAll("[data-filter]").forEach((button) => {
        const selected = button.dataset.filter === "all";
        button.classList.toggle("active", selected);
        button.setAttribute("aria-pressed", String(selected));
      });
    }
    selectCase(linked);
  });
  const initial =
    cases.find(
      (item) => `#${item.dataset.toLowerCase()}/${item.id}` === location.hash,
    ) || cases[0];
  selectCase(initial);
})();
