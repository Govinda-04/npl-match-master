/* NPL Victory Predictor — win probability form handler */

(function () {
  const form       = document.getElementById('win-form');
  const spinner    = document.getElementById('win-spinner');
  const errorDiv   = document.getElementById('win-error');
  const resultSec  = document.getElementById('win-result');

  // Probability text elements
  const lrBat  = document.getElementById('lr-bat-prob');
  const lrBowl = document.getElementById('lr-bowl-prob');
  const rfBat  = document.getElementById('rf-bat-prob');
  const rfBowl = document.getElementById('rf-bowl-prob');

  let winChart = null;

  form.addEventListener('submit', async function (e) {
    e.preventDefault();

    // Reset UI
    spinner.style.display   = 'flex';
    errorDiv.style.display  = 'none';
    resultSec.style.display = 'none';

    const payload = {
      Batting_team:  form.batting_team.value,
      Bowling_team:  form.bowling_team.value,
      city:          form.city.value,
      Runs_left:     Number(form.runs_left.value),
      Balls_left:    Number(form.balls_left.value),
      Wickets_left:  Number(form.wickets_left.value),
      Total_score:   Number(form.total_score.value),
      crr:           Number(form.crr.value),
      rrr:           Number(form.rrr.value),
    };

    try {
      const res  = await fetch('/predict/win', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      // Populate text
      lrBat.textContent  = data.lr.batting_team_prob  + '%';
      lrBowl.textContent = data.lr.bowling_team_prob  + '%';
      rfBat.textContent  = data.rf.batting_team_prob  + '%';
      rfBowl.textContent = data.rf.bowling_team_prob  + '%';

      // Render / update pie chart (LR probabilities)
      const ctx = document.getElementById('win-chart').getContext('2d');
      const labels = [payload.Batting_team, payload.Bowling_team];
      const values = [data.lr.batting_team_prob, data.lr.bowling_team_prob];

      if (winChart) {
        winChart.data.labels          = labels;
        winChart.data.datasets[0].data = values;
        winChart.update();
      } else {
        winChart = new Chart(ctx, {
          type: 'pie',
          data: {
            labels,
            datasets: [{
              data:            values,
              backgroundColor: ['#d4a017', '#2e8b57'],
              borderColor:     ['#f0c040', '#1a5c32'],
              borderWidth:     2,
            }],
          },
          options: {
            responsive:          true,
            maintainAspectRatio: true,
            plugins: {
              legend: {
                labels: { color: '#e8f5e9', font: { size: 13 } },
              },
              tooltip: {
                callbacks: {
                  label: ctx => ` ${ctx.label}: ${ctx.parsed}%`,
                },
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
