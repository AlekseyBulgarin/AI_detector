const textarea = document.getElementById('user_text')
const charCount = document.getElementById('charCount')
const wordCount = document.getElementById('wordCount')
const warningMessage = document.getElementById('warningMessage')
const form = document.getElementById('detectorForm')
const submitBtn = document.getElementById('submitBtn')
const historyKey = 'ai-detector-history'
const preferencesKey = 'ai-detector-preferences'

const viewTitles = {
	analyzer: 'Анализатор', history: 'История', settings: 'Настройки',
	documentation: 'Документация', about: 'О проекте'
}

function initializeIcons() {
	if (window.lucide && typeof window.lucide.createIcons === 'function') window.lucide.createIcons()
}

function updateTextStats() {
	if (!textarea) return
	const text = textarea.value
	if (charCount) charCount.textContent = text.length
	if (wordCount) wordCount.textContent = text.trim() ? text.trim().split(/\s+/).length : 0
	if (!warningMessage) return
	if (text.length > 0 && text.trim().length < 20) {
		warningMessage.textContent = 'Добавьте ещё немного текста для анализа.'
		warningMessage.className = 'input-message is-warning'
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
	function tick(now) {
		const ratio = Math.min((now - startedAt) / 850, 1)
		const eased = 1 - Math.pow(1 - ratio, 3)
		score.textContent = (target * eased).toFixed(1)
		if (ratio < 1) window.requestAnimationFrame(tick)
	}
	window.requestAnimationFrame(tick)
}

function getHistory() {
	try { return JSON.parse(localStorage.getItem(historyKey) || '[]') } catch (error) { return [] }
}

function saveHistory() {
	const resultCard = document.getElementById('resultCard')
	if (!resultCard || !textarea || !textarea.value.trim()) return
	const item = {
		id: document.querySelector('.feedback')?.dataset.analysisId || `local-${Date.now()}`,
		text: textarea.value.trim(),
		probability: Number(resultCard.dataset.probability) || 0,
		createdAt: new Date().toISOString()
	}
	const history = getHistory().filter((entry) => entry.id !== item.id)
	history.unshift(item)
	localStorage.setItem(historyKey, JSON.stringify(history.slice(0, 50)))
}

function formatDate(value) {
	return new Intl.DateTimeFormat('ru-RU', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }).format(new Date(value))
}

function renderHistory() {
	const list = document.getElementById('historyList')
	const count = document.getElementById('historyCount')
	if (!list) return
	const history = getHistory()
	if (count) count.textContent = history.length
	if (!history.length) {
		list.innerHTML = '<div class="empty-state"><i data-lucide="history"></i><strong>История пока пуста</strong><span>Запустите анализ, чтобы сохранить первую проверку на этом устройстве.</span></div>'
		initializeIcons()
		return
	}
	list.innerHTML = history.map((item) => {
		const score = Number(item.probability).toFixed(1)
		const verdict = item.probability > 65 ? 'Высокая вероятность AI' : item.probability > 35 ? 'Смешанные сигналы' : 'Человеческий стиль'
		return `<article class="history-item" data-history-id="${escapeHtml(item.id)}"><div class="history-icon"><i data-lucide="file-text"></i></div><div class="history-content"><strong>${escapeHtml(item.text.slice(0, 92))}${item.text.length > 92 ? '…' : ''}</strong><span>${formatDate(item.createdAt)} · ${verdict}</span></div><div class="history-score">${score}%</div><div class="history-actions"><button type="button" class="history-view" data-history-view="${escapeHtml(item.id)}">Открыть</button><button type="button" class="history-delete" data-history-delete="${escapeHtml(item.id)}" aria-label="Удалить анализ"><i data-lucide="trash-2"></i></button></div></article>`
	}).join('')
	initializeIcons()
}

function escapeHtml(value) {
	return String(value).replace(/[&<>'"]/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#039;', '"': '&quot;' }[character]))
}

function clearHistory() {
	localStorage.removeItem(historyKey)
	renderHistory()
}

function showView(viewName) {
	document.querySelectorAll('.dashboard-view').forEach((view) => view.classList.toggle('is-active', view.dataset.view === viewName))
	document.querySelectorAll('[data-view-target]').forEach((item) => {
		const active = item.dataset.viewTarget === viewName
		item.classList.toggle('is-active', active)
		if (item.classList.contains('nav-item')) item.setAttribute('aria-current', active ? 'page' : 'false')
	})
	const title = document.getElementById('pageTitle')
	if (title) title.textContent = viewTitles[viewName] || viewTitles.analyzer
	document.body.classList.remove('sidebar-open')
	if (viewName === 'history') renderHistory()
}

function loadPreferences() {
	let preferences = {}
	try { preferences = JSON.parse(localStorage.getItem(preferencesKey) || '{}') } catch (error) { preferences = {} }
	const theme = preferences.theme || 'system'
	document.documentElement.dataset.theme = theme
	const themeInput = document.querySelector(`input[name="theme"][value="${theme}"]`)
	if (themeInput) themeInput.checked = true
	const details = document.getElementById('detailsToggle')
	const compact = document.getElementById('compactToggle')
	if (details) { details.checked = preferences.details !== false; document.body.classList.toggle('hide-details', !details.checked) }
	if (compact) { compact.checked = preferences.compact === true; document.body.classList.toggle('compact-mode', compact.checked) }
}

function savePreferences() {
	const theme = document.querySelector('input[name="theme"]:checked')?.value || 'system'
	const details = document.getElementById('detailsToggle')?.checked !== false
	const compact = document.getElementById('compactToggle')?.checked === true
	localStorage.setItem(preferencesKey, JSON.stringify({ theme, details, compact }))
	document.documentElement.dataset.theme = theme
	document.body.classList.toggle('hide-details', !details)
	document.body.classList.toggle('compact-mode', compact)
}

if (textarea) textarea.addEventListener('input', updateTextStats)
if (form) form.addEventListener('submit', function (event) {
	if (!textarea || textarea.value.trim().length < 20) { event.preventDefault(); updateTextStats(); textarea?.focus(); return }
	setLoading(true)
})

document.addEventListener('keydown', function (event) {
	if ((event.metaKey || event.ctrlKey) && event.key === 'Enter' && form && textarea && textarea.value.trim().length >= 20) { event.preventDefault(); form.requestSubmit() }
})

document.addEventListener('DOMContentLoaded', function () {
	initializeIcons()
	updateTextStats()
	loadPreferences()
	if (document.getElementById('resultCard')) { saveHistory(); animateScore() }
	renderHistory()

	document.querySelectorAll('[data-view-target]').forEach((item) => item.addEventListener('click', () => showView(item.dataset.viewTarget)))
	const collapse = document.getElementById('sidebarCollapse')
	const mobileMenu = document.getElementById('mobileMenu')
	const backdrop = document.getElementById('sidebarBackdrop')
	collapse?.addEventListener('click', function () { document.body.classList.toggle('sidebar-collapsed'); collapse.setAttribute('aria-expanded', String(!document.body.classList.contains('sidebar-collapsed'))) })
	mobileMenu?.addEventListener('click', function () { const open = document.body.classList.toggle('sidebar-open'); mobileMenu.setAttribute('aria-expanded', String(open)) })
	backdrop?.addEventListener('click', () => document.body.classList.remove('sidebar-open'))
	document.querySelectorAll('input[name="theme"]').forEach((input) => input.addEventListener('change', savePreferences))
	document.getElementById('detailsToggle')?.addEventListener('change', savePreferences)
	document.getElementById('compactToggle')?.addEventListener('change', savePreferences)
	document.getElementById('clearHistoryTop')?.addEventListener('click', clearHistory)
	document.getElementById('clearHistorySettings')?.addEventListener('click', clearHistory)

	const feedback = document.querySelector('.feedback')
	if (feedback) feedback.querySelectorAll('.feedback-btn').forEach((button) => button.addEventListener('click', async function () {
		feedback.querySelectorAll('.feedback-btn').forEach((item) => { item.disabled = true })
		button.classList.add('is-selected')
		try {
			const response = await fetch('/api/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ analysis_id: feedback.dataset.analysisId, label: button.dataset.label }) })
			if (!response.ok) throw new Error('feedback request failed')
			feedback.querySelector('.feedback-status').textContent = 'Спасибо — ответ сохранён для проверки.'
		} catch (error) {
			feedback.querySelectorAll('.feedback-btn').forEach((item) => { item.disabled = false })
			button.classList.remove('is-selected')
			feedback.querySelector('.feedback-status').textContent = 'Не удалось сохранить ответ. Попробуйте ещё раз.'
		}
	}))

	document.getElementById('historyList')?.addEventListener('click', function (event) {
		const viewId = event.target.closest('[data-history-view]')?.dataset.historyView
		const deleteId = event.target.closest('[data-history-delete]')?.dataset.historyDelete
		if (viewId) {
			const item = getHistory().find((entry) => entry.id === viewId)
			if (item && textarea) { textarea.value = item.text; updateTextStats(); showView('analyzer'); textarea.focus() }
		}
		if (deleteId) { localStorage.setItem(historyKey, JSON.stringify(getHistory().filter((entry) => entry.id !== deleteId))); renderHistory() }
	})
})
