// Multiple-choice quiz widget, shared by every lesson.
//
// Markup:
//   <div class="quiz" data-answer="b">
//     <p class="q">Question?</p>
//     <button data-k="a">Option A</button>
//     <button data-k="b">Option B</button>
//     <p class="why">Explanation shown after answering.</p>
//   </div>
//   <p class="score"></p>   (optional: first-try score for the page)
//
// Options are shuffled on load so their position gives no clue.
// The first click counts towards the score; "Try again" resets the question.

(function () {
  const quizzes = Array.from(document.querySelectorAll(".quiz"));
  const firstTry = new Map(); // quiz -> true/false

  function shuffle(quiz) {
    const buttons = Array.from(quiz.querySelectorAll("button[data-k]"));
    const why = quiz.querySelector(".why");
    for (let i = buttons.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [buttons[i], buttons[j]] = [buttons[j], buttons[i]];
    }
    buttons.forEach((b) => quiz.insertBefore(b, why));
  }

  function updateScore() {
    const el = document.querySelector(".score");
    if (!el) return;
    const right = Array.from(firstTry.values()).filter(Boolean).length;
    el.textContent =
      firstTry.size === quizzes.length
        ? `First-try score: ${right} / ${quizzes.length}. Questions you missed are the ones to revisit in a few days.`
        : `Answered ${firstTry.size} of ${quizzes.length}.`;
  }

  quizzes.forEach((quiz) => {
    shuffle(quiz);
    const answer = quiz.dataset.answer;
    const buttons = Array.from(quiz.querySelectorAll("button[data-k]"));

    const retry = document.createElement("button");
    retry.className = "retry";
    retry.type = "button";
    retry.textContent = "Try again";
    quiz.appendChild(retry);

    buttons.forEach((b) =>
      b.addEventListener("click", () => {
        const ok = b.dataset.k === answer;
        if (!firstTry.has(quiz)) firstTry.set(quiz, ok);
        b.classList.add(ok ? "correct" : "wrong");
        if (ok) buttons.forEach((x) => (x.disabled = true));
        quiz.classList.add("answered");
        updateScore();
      })
    );

    retry.addEventListener("click", () => {
      buttons.forEach((x) => {
        x.disabled = false;
        x.classList.remove("correct", "wrong");
      });
      quiz.classList.remove("answered");
      shuffle(quiz);
    });
  });

  updateScore();
})();
