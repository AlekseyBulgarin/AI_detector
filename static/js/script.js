// Подсчёт символов в реальном времени
const textarea = document.getElementById('user_text')
const charCount = document.getElementById('charCount')
const warningMessage = document.getElementById('warningMessage')
const form = document.getElementById('detectorForm')
const submitBtn = document.getElementById('submitBtn')
const btnText = submitBtn?.querySelector('.btn-text')
const spinner = submitBtn?.querySelector('.spinner')

// Обновление счётчика символов
function updateCharCount() {
	if (!textarea || !charCount) return

	const length = textarea.value.length
	charCount.textContent = length

	if (length > 0 && length < 20) {
		warningMessage.textContent = '⚠️ Минимум 20 символов для анализа'
		warningMessage.style.color = '#f59e0b'
	} else if (length >= 20) {
		warningMessage.textContent = '✅ Достаточно для анализа'
		warningMessage.style.color = '#10b981'
	} else {
		warningMessage.textContent = ''
	}
}

// Показ загрузки при отправке
if (form) {
	form.addEventListener('submit', function () {
		if (submitBtn) {
			if (btnText) btnText.textContent = 'Анализируем...'
			if (spinner) spinner.classList.remove('hidden')
			submitBtn.disabled = true
		}
	})
}

// Автоматическое расширение textarea при вводе
if (textarea) {
	textarea.addEventListener('input', function () {
		this.style.height = 'auto'
		this.style.height = Math.min(this.scrollHeight, 300) + 'px'
		updateCharCount()
	})

	updateCharCount()
	textarea.dispatchEvent(new Event('input'))
}

// Очистка textarea от пробелов при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    const textarea = document.getElementById('user_text');
    if (textarea) {
        // Если в поле только пробелы или пустота — очищаем
        if (!textarea.value.trim() || textarea.value.trim() === '') {
            textarea.value = '';
        }
        // Обновляем счётчик символов (если он есть)
        if (typeof updateCharCount === 'function') {
            updateCharCount();
        }
    }
});
