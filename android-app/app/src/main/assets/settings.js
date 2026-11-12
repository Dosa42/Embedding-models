const byId = id => document.getElementById(id)
const dialog = byId('settings-dialog')
const fields = [
  {id: 'size', key: 'max_message_size', scale: 1048576, integer: true},
  {id: 'queue', key: 'max_queue', scale: 1, integer: true},
  {id: 'interval', key: 'ping_interval', scale: 1},
  {id: 'timeout', key: 'ping_timeout', scale: 1}
]
let refreshTimer = null
let loading = false
let dirty = false

function status(message, error = false) {
  byId('settings-status').textContent = message
  byId('settings-status').classList.toggle('settings-error', error)
}

async function api(body) {
  const response = await fetch('/settings/websocket', body ? {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {})
  const result = await response.json()
  if (!response.ok) {
    const detail = result.detail
    throw new Error(Array.isArray(detail) ? detail.map(item => `${item.loc.slice(1).join(' · ')}: ${item.msg}`).join('; ') : detail || 'Could not update connection settings')
  }
  return result
}

function updateModes() {
  for (const field of fields) byId(`ws-${field.id}-value`).disabled = byId(`ws-${field.id}-mode`).value !== 'manual'
}

function populate(saved) {
  byId('ws-enabled').checked = saved.enabled
  for (const field of fields) {
    byId(`ws-${field.id}-mode`).value = saved[field.key].mode
    byId(`ws-${field.id}-value`).value = saved[field.key].value === null ? '' : saved[field.key].value / field.scale
  }
  byId('ws-compression').value = saved.compression
  byId('ws-refresh').value = saved.adaptive_refresh_seconds
  updateModes()
  dirty = false
}

function renderRunning(result) {
  const value = result.effective
  const format = number => Number(number).toLocaleString(undefined, {maximumFractionDigits: 3})
  const rows = [
    ['Transport', value.ws === 'none' ? 'Disabled' : 'Enabled'],
    ['Message size', value.ws_max_size === null ? 'Unlimited' : `${format(value.ws_max_size / 1048576)} MiB`],
    ['Queue', value.ws_max_queue === null ? 'Unlimited' : `${format(value.ws_max_queue)} messages`],
    ['Ping interval', value.ws_ping_interval === null ? 'Disabled' : `${format(value.ws_ping_interval)} seconds`],
    ['Pong timeout', value.ws_ping_timeout === null ? 'Disabled' : `${format(value.ws_ping_timeout)} seconds`],
    ['Compression', value.ws_per_message_deflate ? 'Enabled' : 'Disabled']
  ]
  byId('ws-effective').replaceChildren(...rows.flatMap(([label, content]) => {
    const term = document.createElement('dt')
    const description = document.createElement('dd')
    term.textContent = label
    description.textContent = content
    return [term, description]
  }))
  byId('ws-connections').textContent = `${result.active_connections} open ${result.active_connections === 1 ? 'socket' : 'sockets'}`
  const device = result.device
  byId('ws-device').textContent = `${format(device.available_memory_bytes / 1048576)} MiB available · ${device.power_save ? 'Power saving on' : 'Power saving off'} · ${device.interactive ? 'Screen active' : 'Screen inactive'}`
}

function readPositive(input, label, integer = false, scale = 1) {
  const value = Number(input.value) * scale
  if (!input.value.trim() || !Number.isFinite(value) || value <= 0 || (integer && !Number.isSafeInteger(value))) {
    input.setAttribute('aria-invalid', 'true')
    input.focus()
    throw new Error(`Enter a positive ${integer ? 'whole number' : 'number'} for ${label}.`)
  }
  input.removeAttribute('aria-invalid')
  return value
}

function collect() {
  const result = {enabled: byId('ws-enabled').checked, compression: byId('ws-compression').value, adaptive_refresh_seconds: readPositive(byId('ws-refresh'), 'adaptive refresh')}
  for (const field of fields) {
    const mode = byId(`ws-${field.id}-mode`).value
    const input = byId(`ws-${field.id}-value`)
    result[field.key] = {mode, value: mode === 'manual' ? readPositive(input, field.key.replaceAll('_', ' '), field.integer, field.scale) : null}
  }
  return result
}

async function reload() {
  if (loading) return
  loading = true
  byId('save-settings').disabled = true
  byId('reload-settings').disabled = true
  status('Loading saved settings…')
  try {
    const result = await api()
    populate(result.saved)
    renderRunning(result)
    status('Saved choices loaded. Changes apply when you press Apply settings.')
  } catch (error) {
    status(error.message, true)
  } finally {
    loading = false
    byId('save-settings').disabled = false
    byId('reload-settings').disabled = false
  }
}

byId('open-settings').onclick = async () => {
  dialog.showModal()
  byId('settings-title').focus()
  await reload()
  if (!dialog.open) return
  refreshTimer = setInterval(async () => {
    if (loading || !dialog.open) return
    try { renderRunning(await api()) } catch (error) { if (!dirty) status(error.message, true) }
  }, 3000)
}

byId('close-settings').onclick = () => dialog.close()
dialog.addEventListener('keydown', event => {
  if (event.key !== 'Tab') return
  const controls = Array.from(dialog.querySelectorAll('button:not([disabled]),input:not([disabled]),select:not([disabled]),summary')).filter(control => control.getClientRects().length)
  const first = controls[0]
  const last = controls.at(-1)
  if (event.shiftKey && (document.activeElement === first || document.activeElement === byId('settings-title'))) {
    event.preventDefault()
    last.focus()
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault()
    first.focus()
  }
})
dialog.addEventListener('close', () => {
  clearInterval(refreshTimer)
  refreshTimer = null
  byId('open-settings').focus()
})
byId('reload-settings').onclick = reload
byId('settings-form').addEventListener('input', event => {
  event.target.removeAttribute('aria-invalid')
  dirty = true
  status('Unsaved changes')
})
byId('settings-form').addEventListener('change', updateModes)
byId('settings-form').onsubmit = async event => {
  event.preventDefault()
  if (loading) return
  let body
  try { body = collect() } catch (error) { status(error.message, true); return }
  loading = true
  byId('save-settings').disabled = true
  byId('reload-settings').disabled = true
  status('Applying your choices…')
  try {
    const result = await api(body)
    populate(result.saved)
    renderRunning(result)
    status('Applied and saved on this device.')
  } catch (error) {
    status(error.message, true)
  } finally {
    loading = false
    byId('save-settings').disabled = false
    byId('reload-settings').disabled = false
  }
}
