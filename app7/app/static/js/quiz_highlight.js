document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('input[type=radio]').forEach(radio => {
      radio.addEventListener('change', function() {
        const name = this.name;
        document.querySelectorAll(`input[name="${name}"]`).forEach(input => {
          input.parentElement.classList.remove('selected');
        });
        this.parentElement.classList.add('selected');
      });
    });
  });
  