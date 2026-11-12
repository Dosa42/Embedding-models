const $ = id => document.getElementById(id)
let folderFiles = []

function updateContext() {
  $('active-project').textContent = $('project').selectedOptions[0]?.textContent || 'Your project'
  $('active-model').textContent = $('model').selectedOptions[0]?.textContent || 'E5 multilingual'
}

function resetSearch() {
  updateContext()
  $('results').replaceChildren()
  $('search-empty').hidden = false
  $('result-count').textContent = 'Ready to explore'
  $('search-status').textContent = 'Ask a question to explore this project’s indexed source.'
}

function setProgress(value) {
  $('progress-fill').style.width = `${value}%`
  $('progress-fill').parentElement.setAttribute('aria-valuenow', String(value))
}

async function request(path, body) {
  const response = await fetch(path, body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {})
  const data = await response.json()
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data))
  return data
}

const library = {
  projects: () => request('/projects'),
  createProject: name => request('/projects', {name}),
  add: documents => request('/documents', {project: $('project').value, model: $('model').value, documents}),
  search: query => request('/search', {project: $('project').value, model: $('model').value, query}),
  folderFiles: () => request('/folder/files'),
  corpusFiles: () => request('/corpus/files'),
  readFile: file => request('/' + file.source + '/file/' + encodeURIComponent(file.id))
}

async function refreshProjects() {
  const data = await library.projects()
  const current = $('project').value
  $('project').replaceChildren()
  for (const project of data.projects) {
    const option = document.createElement('option')
    option.value = project.id
    option.textContent = project.name
    $('project').append(option)
  }
  if (current) $('project').value = current
  updateContext()
  if (!data.projects.length) {
    await library.createProject('My project')
    await refreshProjects()
  }
}


function chunks(path, content) {
  const lines = content.replaceAll('\r\n', '\n').split('\n')
  const result = []
  for (let start = 0; start < lines.length; start += 28) {
    const text = lines.slice(start, start + 34).join('\n')
    if (text.trim()) result.push({path, line: start + 1, text})
  }
  return result
}

async function saveItems(items) {
  await library.add(items)
  setProgress(100)
  $('ingest-status').textContent = `${items.length} passages saved to Chroma`
}

$('new-project').onclick = async () => {
  const name = prompt('Project name')?.trim()
  if (!name) return
  try {
    const project = await library.createProject(name)
    await refreshProjects()
    $('project').value = project.id
    resetSearch()
  } catch (error) { $('ingest-status').textContent = error.message }
}

$('files').onchange = () => {
  folderFiles = []
  const files = Array.from($('files').files)
  $('selected').textContent = files.length ? `${files.length} file${files.length === 1 ? '' : 's'} selected` : 'No files selected'
}

$('folder').onclick = () => {
  if (!window.FolderPicker) { $('ingest-status').textContent = 'Folder picker unavailable'; return }
  $('selected').textContent = 'Reading folder…'
  window.FolderPicker.pickFolder()
}

window.onFolderReady = async () => {
  try {
    folderFiles = (await library.folderFiles()).files.map(file => ({...file, source: 'folder'}))
    $('files').value = ''
    $('selected').textContent = `${folderFiles.length} files in source folder`
  } catch (error) { $('selected').textContent = error.message }
}

$('corpus').onclick = async () => {
  try {
    folderFiles = (await library.corpusFiles()).files.map(file => ({...file, source: 'corpus'}))
    $('files').value = ''
    $('selected').textContent = `${folderFiles.length} bundled source files selected`
  } catch (error) { $('selected').textContent = error.message }
}

$('ingest').onclick = async () => {
  const files = Array.from($('files').files)
  if (!files.length && !folderFiles.length) { $('ingest-status').textContent = 'Choose source files or a folder first.'; return }
  $('ingest').disabled = true
  $('model').disabled = true
  $('project').disabled = true
  setProgress(0)
  try {
    const documents = []
    let processed = 0
    const total = files.length + folderFiles.length
    async function addSource(path, content) {
      if (content.includes('\0')) return
      for (const item of chunks(path, content)) {
        documents.push(item)
      }
      processed++
      setProgress(Math.round(100 * processed / total))
    }
    for (const file of files) {
      $('ingest-status').textContent = `Reading ${file.name}…`
      await addSource(file.webkitRelativePath || file.name, await file.text())
    }
    for (const file of folderFiles) {
      $('ingest-status').textContent = `Reading ${file.path}…`
      const source = await library.readFile(file)
      await addSource(source.path, source.content)
    }
    if (!documents.length) throw new Error('No nonempty passages found')
    $('ingest-status').textContent = `Embedding and saving ${documents.length} passages…`
    const result = await library.add(documents)
    setProgress(100)
    $('ingest-status').textContent = `${result.saved} passages indexed in ${total} files.`
  } catch (error) { $('ingest-status').textContent = error.message }
  finally { $('ingest').disabled = false; $('model').disabled = false; $('project').disabled = false }
}

$('vectors').onchange = async () => {
  const file = $('vectors').files[0]
  if (!file) return
  $('model').disabled = true
  $('project').disabled = true
  try {
    const rows = file.size ? (await file.text()).split(/\r?\n/).filter(Boolean) : []
    const items = rows.map((row, index) => {
      const entry = JSON.parse(row)
      const vector = entry.vector || entry.embedding
      if (!Array.isArray(vector) || vector.length !== 384) throw new Error(`Line ${index + 1}: expected 384-dimensional vector`)
      return {path: entry.path || file.name, line: entry.line || 1, text: entry.text || entry.content || '', vector}
    })
    if (!items.length) throw new Error('No vectors found')
    await saveItems(items)
    $('ingest-status').textContent = `${items.length} existing vectors saved.`
  } catch (error) { $('ingest-status').textContent = error.message }
  finally { $('model').disabled = false; $('project').disabled = false; $('vectors').value = '' }
}

$('search').onclick = async () => {
  const query = $('query').value.trim()
  if (!query) { $('query').focus(); return }
  $('search').disabled = true
  $('model').disabled = true
  $('project').disabled = true
  $('results').replaceChildren()
  $('search-empty').hidden = true
  $('result-count').textContent = 'Searching…'
  $('search-status').textContent = 'Embedding query and searching Chroma…'
  try {
    const data = await library.search(query)
    $('result-count').textContent = `${data.hits.length} ${data.hits.length === 1 ? 'passage' : 'passages'}`
    $('search-status').textContent = data.hits.length ? `Showing source passages for “${query}”` : 'No passages found in this project and model. Add source in the workspace, or try another model.'
    for (const [index, hit] of data.hits.entries()) {
      const card = document.createElement('article')
      card.className = 'hit'
      const head = document.createElement('div')
      head.className = 'hit-head'
      const rank = document.createElement('span')
      rank.className = 'hit-index'
      rank.textContent = String(index + 1).padStart(2, '0')
      const locator = document.createElement('div')
      locator.className = 'hit-locator'
      const path = document.createElement('strong')
      path.className = 'hit-path'
      path.textContent = hit.citation
      const meta = document.createElement('span')
      meta.className = 'hit-meta'
      meta.textContent = `Source passage · line ${hit.line || String(hit.citation).split(':').pop()}`
      const copy = document.createElement('button')
      copy.className = 'copy-cite'
      copy.type = 'button'
      copy.textContent = 'Copy citation'
      copy.setAttribute('aria-label', `Copy citation ${hit.citation}`)
      copy.onclick = async () => {
        try {
          await navigator.clipboard.writeText(hit.citation)
          copy.textContent = 'Copied'
        } catch (error) { copy.textContent = 'Copy failed' }
      }
      const code = document.createElement('div')
      code.className = 'hit-code'
      const gutter = document.createElement('pre')
      gutter.className = 'line-gutter'
      gutter.setAttribute('aria-hidden', 'true')
      const firstLine = Number(hit.line) || Number(String(hit.citation).split(':').pop()) || 1
      gutter.textContent = String(hit.text).split('\n').map((_, offset) => firstLine + offset).join('\n')
      const snippet = document.createElement('pre')
      snippet.textContent = hit.text
      locator.append(path, meta)
      head.append(rank, locator, copy)
      code.append(gutter, snippet)
      const foot = document.createElement('div')
      foot.className = 'hit-footer'
      foot.textContent = `Semantic distance ${Number(hit.distance).toFixed(3)} · lower is closer`
      card.append(head, code, foot)
      $('results').append(card)
    }
  } catch (error) { $('result-count').textContent = 'Search failed'; $('search-status').textContent = error.message }
  finally { $('search').disabled = false; $('model').disabled = false; $('project').disabled = false }
}

$('query').onkeydown = event => { if (event.key === 'Enter') $('search').click() }
$('project').onchange = resetSearch
$('model').onchange = resetSearch
for (const example of document.querySelectorAll('[data-example]')) {
  example.onclick = () => { $('query').value = example.dataset.example; $('search').click() }
}
refreshProjects().catch(error => { $('ingest-status').textContent = error.message })
