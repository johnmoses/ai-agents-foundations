document.addEventListener('DOMContentLoaded', () => {
    document.querySelector('form').addEventListener('submit', function(event) {
      if (!confirm("Are you sure you want to submit your answers?")) {
        event.preventDefault();
        return false;
      }
    });
  });
  