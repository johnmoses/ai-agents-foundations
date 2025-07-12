document.addEventListener('DOMContentLoaded', () => {
    document.querySelector('form').addEventListener('submit', function(event) {
      const questions = parseInt(this.dataset.questionsCount, 10);
      for (let i = 0; i < questions; i++) {
        const options = document.getElementsByName('q' + i);
        let answered = false;
        for (const option of options) {
          if (option.checked) {
            answered = true;
            break;
          }
        }
        if (!answered) {
          alert(`Please answer question ${i + 1}`);
          event.preventDefault();
          return false;
        }
      }
    });
  });
  