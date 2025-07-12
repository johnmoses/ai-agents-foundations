document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('form');
    const questionsCount = parseInt(form.dataset.questionsCount, 10);
  
    // Load saved answers on page load
    for (let i = 0; i < questionsCount; i++) {
      const saved = localStorage.getItem('quiz_q' + i);
      if (saved) {
        const option = document.querySelector(`input[name="q${i}"][value="${saved}"]`);
        if (option) option.checked = true;
      }
    }
  
    // Save answer on change
    form.querySelectorAll('input[type=radio]').forEach(radio => {
      radio.addEventListener('change', () => {
        localStorage.setItem(radio.name, radio.value);
      });
    });
  
    // Clear saved answers on submit
    form.addEventListener('submit', () => {
      for (let i = 0; i < questionsCount; i++) {
        localStorage.removeItem('quiz_q' + i);
      }
    });
  });
  