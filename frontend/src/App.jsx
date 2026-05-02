import { Routes, Route, Navigate } from 'react-router-dom'
import { QueryProvider } from './context/QueryContext.jsx'
import { GraphProvider } from './context/GraphContext.jsx'
import { IndexingProvider } from './context/IndexingContext.jsx'

// Pages — lazy imported to keep initial bundle small
import { lazy, Suspense } from 'react'

const QueryPage    = lazy(() => import('./pages/QueryPage.jsx'))
const GraphPage    = lazy(() => import('./pages/GraphPage.jsx'))
const EvaluationPage = lazy(() => import('./pages/EvaluationPage.jsx'))
const IndexingPage = lazy(() => import('./pages/IndexingPage.jsx'))

// Navbar and Spinner exist after Stage 2 — stub fallback for Stage 1
function PageLoader() {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: '60vh',
      color: 'var(--text-muted)',
      fontFamily: 'var(--font-mono)',
      fontSize: '0.875rem',
    }}>
      loading…
    </div>
  )
}

export default function App() {
  return (
    <IndexingProvider>
      <GraphProvider>
        <QueryProvider>
          {/* Navbar will be added in Stage 2 */}
          <main className="app-main">
            <Suspense fallback={<PageLoader />}>
              <Routes>
                <Route path="/"            element={<Navigate to="/query" replace />} />
                <Route path="/query"       element={<QueryPage />} />
                <Route path="/graph"       element={<GraphPage />} />
                <Route path="/evaluation"  element={<EvaluationPage />} />
                <Route path="/indexing"    element={<IndexingPage />} />
                <Route path="*"            element={<Navigate to="/query" replace />} />
              </Routes>
            </Suspense>
          </main>
        </QueryProvider>
      </GraphProvider>
    </IndexingProvider>
  )
}