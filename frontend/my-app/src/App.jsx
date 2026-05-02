// src/App.jsx
import { Routes, Route, Navigate } from "react-router-dom"
import { lazy, Suspense } from "react"
import { QueryProvider }    from "./context/QueryContext.jsx"
import { GraphProvider }    from "./context/GraphContext.jsx"
import { IndexingProvider } from "./context/IndexingContext.jsx"
import { ToastProvider }    from "./components/common/Toast.jsx"
import Navbar               from "./components/common/Navbar.jsx"
import { PageSpinner }      from "./components/common/Spinner.jsx"

const QueryPage      = lazy(() => import("./pages/QueryPage.jsx"))
const GraphPage      = lazy(() => import("./pages/GraphPage.jsx"))
const EvaluationPage = lazy(() => import("./pages/EvaluationPage.jsx"))
const IndexingPage   = lazy(() => import("./pages/IndexingPage.jsx"))

export default function App() {
  return (
    <IndexingProvider>
      <GraphProvider>
        <QueryProvider>
          <ToastProvider>
            <div className="app-shell">
              <Navbar />
              <main className="app-main">
                <Suspense fallback={<PageSpinner label="Loading page…" />}>
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
            </div>
          </ToastProvider>
        </QueryProvider>
      </GraphProvider>
    </IndexingProvider>
  )
}