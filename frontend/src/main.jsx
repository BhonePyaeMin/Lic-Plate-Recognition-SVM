import React from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource-variable/dm-sans'
import './styles/tokens.css'
import './styles/base.css'
import './styles/components.css'
import './styles/prepare.css'
import './styles/svm.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
