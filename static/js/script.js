const textarea = document.getElementById('user_text')
const charCount = document.getElementById('charCount')
const wordCount = document.getElementById('wordCount')
const warningMessage = document.getElementById('warningMessage')
const form = document.getElementById('detectorForm')
const submitBtn = document.getElementById('submitBtn')

function updateTextStats() {
	if (!textarea) return

	const text = textarea.value
	const characters = text.length
	const words = text.trim() ? text.trim().split(/\s+/).length : 0
	if (charCount) charCount.textContent = characters
	if (wordCount) wordCount.textContent = words

	if (!warningMessage) return
	if (characters > 0 && characters < 20) {
		warningMessage.textContent = 'Добавьте ещё немного текста для точного анализа.'
		warningMessage.className = 'input-message is-warning'
	} else if (characters >= 20) {
		warningMessage.textContent = ''
		warningMessage.className = 'input-message'
	} else {
		warningMessage.textContent = ''
		warningMessage.className = 'input-message'
	}
}

function setLoading(isLoading) {
	if (!submitBtn) return
	submitBtn.disabled = isLoading
	submitBtn.classList.toggle('is-loading', isLoading)
}

function animateScore() {
	const resultCard = document.getElementById('resultCard')
	const score = document.getElementById('animatedScore')
	const progress = document.getElementById('progressFill')
	if (!resultCard || !score || !progress) return

	const target = Math.max(0, Math.min(100, Number(resultCard.dataset.probability) || 0))
	progress.style.setProperty('--score', `${target / 100}`)
	progress.classList.add('is-animated')

	const startedAt = performance.now()
	const duration = 900
	function tick(now) {
		const progressRatio = Math.min((now - startedAt) / duration, 1)
		const eased = 1 - Math.pow(1 - progressRatio, 3)
		score.textContent = (target * eased).toFixed(1)
		if (progressRatio < 1) window.requestAnimationFrame(tick)
	}
	window.requestAnimationFrame(tick)
}

function initializeIcons() {
	if (window.lucide && typeof window.lucide.createIcons === 'function') {
		window.lucide.createIcons()
	}
}

if (textarea) {
	textarea.addEventListener('input', updateTextStats)
	updateTextStats()
}

if (form) {
	form.addEventListener('submit', function (event) {
		if (!textarea || textarea.value.trim().length < 20) {
			event.preventDefault()
			updateTextStats()
			textarea?.focus()
			return
		}
		setLoading(true)
	})
}

document.addEventListener('keydown', function (event) {
	if ((event.metaKey || event.ctrlKey) && event.key === 'Enter' && form && textarea) {
		event.preventDefault()
		if (textarea.value.trim().length >= 20) form.requestSubmit()
	}
})

document.addEventListener('DOMContentLoaded', function () {
	initializeIcons()
	animateScore()

	const feedback = document.querySelector('.feedback')
	if (!feedback) return

	const status = feedback.querySelector('.feedback-status')
	feedback.querySelectorAll('.feedback-btn').forEach(function (button) {
		button.addEventListener('click', async function () {
			feedback.querySelectorAll('.feedback-btn').forEach((item) => { item.disabled = true })
			button.classList.add('is-selected')
			try {
				const response = await fetch('/api/feedback', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ analysis_id: feedback.dataset.analysisId, label: button.dataset.label })
				})
				if (!response.ok) throw new Error('feedback request failed')
				if (status) status.textContent = 'Спасибо — ответ сохранён для проверки.'
			} catch (error) {
				feedback.querySelectorAll('.feedback-btn').forEach((item) => { item.disabled = false })
				button.classList.remove('is-selected')
				if (status) status.textContent = 'Не удалось сохранить ответ. Попробуйте ещё раз.'
			}
		})
	})
})
