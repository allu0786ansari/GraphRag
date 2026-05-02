// src/utils/constants.js

export const COMMUNITY_LEVELS = [
  {
    value: 'c0',
    label: 'C0 — Global',
    description: 'Broadest view. 3–5 top-level themes. Cheapest — fewest LLM calls.',
    color: '#f0a500',
    shortLabel: 'C0',
  },
  {
    value: 'c1',
    label: 'C1 — Regional',
    description: 'Balanced. 15–25 communities. Default choice for most queries.',
    color: '#d48900',
    shortLabel: 'C1',
  },
  {
    value: 'c2',
    label: 'C2 — Local',
    description: 'Detailed. 50–100 communities. Better for specific sub-topics.',
    color: '#a86b00',
    shortLabel: 'C2',
  },
  {
    value: 'c3',
    label: 'C3 — Granular',
    description: 'Most detailed. 150–300 communities. Expensive — many LLM calls.',
    color: '#7a4d00',
    shortLabel: 'C3',
  },
]

export const SAMPLE_QUESTIONS = [
  {
    text: 'What are the main themes and topics covered across the entire document corpus?',
    category: 'thematic',
  },
  {
    text: 'Who are the most influential organizations and what roles do they play in shaping the field?',
    category: 'entity',
  },
  {
    text: 'What are the key trends and developments that have emerged over time in this domain?',
    category: 'temporal',
  },
  {
    text: 'What are the major points of disagreement or tension between different perspectives in the corpus?',
    category: 'comparative',
  },
  {
    text: 'What are the most significant challenges and open problems described across the documents?',
    category: 'analytical',
  },
  {
    text: 'How do the different communities of interest relate to and influence each other?',
    category: 'relational',
  },
  {
    text: 'What opportunities or positive developments are highlighted throughout the corpus?',
    category: 'analytical',
  },
  {
    text: 'What recommendations or calls to action appear most frequently across the documents?',
    category: 'prescriptive',
  },
]

export const EVAL_CRITERIA = [
  {
    id: 'comprehensiveness',
    label: 'Comprehensiveness',
    description: 'Does the answer cover all relevant aspects of the question?',
    icon: 'Layers',
  },
  {
    id: 'diversity',
    label: 'Diversity',
    description: 'Does the answer present multiple perspectives and viewpoints?',
    icon: 'GitBranch',
  },
  {
    id: 'empowerment',
    label: 'Empowerment',
    description: 'Does the answer help the reader understand and make decisions?',
    icon: 'Zap',
  },
  {
    id: 'directness',
    label: 'Directness',
    description: 'Is the answer specific, well-organized and easy to follow?',
    icon: 'Target',
  },
]

export const PIPELINE_STAGES = [
  { id: 'chunking',            label: 'Chunking',            description: 'Split documents into 600-token chunks' },
  { id: 'extraction',          label: 'Extraction',          description: 'Extract entities and relationships via LLM' },
  { id: 'graph_construction',  label: 'Graph Build',         description: 'Deduplicate and build knowledge graph' },
  { id: 'community_detection', label: 'Communities',         description: 'Leiden hierarchical community detection' },
  { id: 'summarization',       label: 'Summarization',       description: 'LLM summary per community' },
  { id: 'embedding',           label: 'Embeddings + FAISS',  description: 'Embed chunks and build vector index' },
]

export const ENTITY_TYPE_COLORS = {
  PERSON:       '#f0a500',
  ORGANIZATION: '#1fb899',
  LOCATION:     '#3b82f6',
  CONCEPT:      '#a78bfa',
  TECHNOLOGY:   '#f472b6',
  EVENT:        '#fb923c',
  PRODUCT:      '#34d399',
  DEFAULT:      '#6b7280',
}

export const CATEGORY_COLORS = {
  thematic:     { bg: 'rgba(240,165,0,0.1)',   text: '#f0a500',  border: 'rgba(240,165,0,0.25)' },
  entity:       { bg: 'rgba(31,184,153,0.1)',  text: '#1fb899',  border: 'rgba(31,184,153,0.25)' },
  temporal:     { bg: 'rgba(59,130,246,0.1)',  text: '#3b82f6',  border: 'rgba(59,130,246,0.25)' },
  comparative:  { bg: 'rgba(167,139,250,0.1)', text: '#a78bfa',  border: 'rgba(167,139,250,0.25)' },
  analytical:   { bg: 'rgba(244,114,182,0.1)', text: '#f472b6',  border: 'rgba(244,114,182,0.25)' },
  relational:   { bg: 'rgba(251,146,60,0.1)',  text: '#fb923c',  border: 'rgba(251,146,60,0.25)' },
  prescriptive: { bg: 'rgba(52,211,153,0.1)',  text: '#34d399',  border: 'rgba(52,211,153,0.25)' },
}

export const API_KEY = import.meta.env.VITE_API_KEY || ''