// Vendor bundle entry point.
//
// build.mjs bundles this file (with its npm imports) into dist/vendor.<hash>.js
// as a single IIFE that re-exposes the libraries on `window`. The app's
// classic-script .jsx files (each wrapped in its own IIFE) read these globals
// exactly as they did when the libraries came from CDN <script> tags:
//
//   window.React      -> React namespace (createElement, Fragment, hooks, ...)
//   window.ReactDOM   -> ReactDOM (createRoot)
//   window.marked     -> marked() with .parse()
//   window.DOMPurify  -> DOMPurify with .sanitize()
//
// Keeping this a separate bundle (rather than inlining into app.<hash>.js)
// means the vendor bytes are cached independently of app-code changes.

import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { marked } from 'marked';
import DOMPurify from 'dompurify';

// Model-generated Markdown must never make the browser issue a remote request
// (an injected source document can steer the model to emit e.g.
// <img src="https://evil/…">). Forbid remote-loading and form elements globally —
// the app never renders them.
DOMPurify.setConfig({
  FORBID_TAGS: [
    'img', 'iframe', 'video', 'audio', 'source', 'track', 'picture',
    'object', 'embed', 'form', 'input', 'button', 'select', 'textarea',
  ],
});

window.React = React;
window.ReactDOM = ReactDOM;
window.marked = marked;
window.DOMPurify = DOMPurify;
