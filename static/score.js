/* NPL Score Predictor — score prediction form handler */

(function () {
  const form      = document.getElementById('score-form');
  const spinner   = document.getElementById('score-spinner');
  const errorDiv  = document.getElementById('score-error');
  const resultSec = document.getElementById('score-result');
  const scoreVal  = document.getElementById('predicted-score-value');

  let scoreChart = null;

  form.addEventListener('submit', async function (e) {
    e.preventDefault();

    // Reset UI
    spinner.style.display   = 'flex';
    errorDiv.style.display  = 'none';
    resultSec.style.display = 'none';

    const payload = {
      batting_team: form.batting_team.value,
      bowling_team: form.bowling_team.value,
      venue:        form.venue.value,
      city:         form.city.value,
      cur_score:    Number(form.cur_score.value),
      crr:          Number(form.crr.value),
      balls_left:   Number(form.balls_left.value),
      wicket_left:  Number(form.wicket_left.value),
      last_5_ov:    Number(form.last_5_ov.value),
    };

    try {
      const res  = await fetch('/predict/score', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      const predicted = data.predicted_score;
      scoreVal.textContent = predicted;

      // Render / update bar chart
      const ctx    = document.getElementById('score-chart').getContext('2d');
      const labels = ['Current Score', 'Predicted Final Score'];
      const values = [payload.cur_score, predicted];

      if (scoreChart) {
        scoreChart.data.datasets[0].data = values;
        scoreChart.update();
      } else {
        scoreChart = new Chart(ctx, {
          type: 'bar',
          data: {
            labels,
            datasets: [{
              label:           'Runs',
              data:            values,
              backgroundColor: ['#2e8b57', '#d4a017'],
              borderColor:     ['#1a5c32', '#f0c040'],
              borderWidth:     2,
              borderRadius:    6,
            }],
          },
          options: {
            responsive:          true,
            maintainAspectRatio: true,
            plugins: {
              legend: { display: false },
              tooltip: {
                callbacks: {
                  label: ctx => ` ${ctx.parsed.y} runs`,
                },
              },
            },
            scales: {
              x: {
                ticks: { color: '#e8f5e9' },
                grid:  { color: 'rgba(46,139,87,0.3)' },
              },
              y: {
                beginAtZero: true,
                ticks: { color: '#e8f5e9' },
                grid:  { color: 'rgba(46,139,87,0.3)' },
              },
            },
          },
        });
      }

      spinner.style.display   = 'none';
      resultSec.style.display = 'flex';

    } catch (err) {
      spinner.style.display  = 'none';
      errorDiv.textContent   = err.message;
      errorDiv.style.display = 'block';
    }
  });
})();
