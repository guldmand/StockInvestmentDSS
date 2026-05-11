# Frontend

## Purpose

The frontend is the minimal browser-based entry point for the StockInvestmentDSS V1.0 PoC.

It belongs to the application track and is intentionally static for now: HTML, compiled CSS, and vanilla JavaScript. The goal is to prove the local DSS flow before adding heavier product UI, login flows, charts, or framework-based frontend architecture.

## Current scope

The current frontend covers issue `#4 Create minimal web app shell`.

Implemented:

- Static web app shell in `system/frontend/public/index.html`
- Vanilla JavaScript integration in `system/frontend/public/js/app.js`
- SCSS source files in `system/frontend/src/scss/`
- Compiled CSS files in `system/frontend/public/css/`
- Backend health status block
- Runtime configuration block
- Placeholder cards for DSS concepts
- Links to backend OpenAPI/Swagger resources

The frontend does not:

- Run model training
- Access DuckDB directly
- Access guldNAS directly
- Depend on React, Next.js, FinRL, or external cloned repos
- Implement login yet

Login/front-page verification belongs to issue `#39`.

Frontend containerization belongs to issue `#163`.

## Runtime assumptions

During local development, the frontend is served separately from the backend.

Example local frontend server:

```bash
cd system/frontend/public
python -m http.server 3001 --bind 127.0.0.1
```

Example backend startup:

```bash
cd system
docker compose up --build
```

The frontend calls the backend API through `app.js`.

Typical API base URL:

```js
const API_BASE_URL = "http://localhost:8000";
```

If the backend is exposed on another host port, this value must match the Docker Compose port mapping.

## Backend endpoints used

The frontend currently calls:

```txt
GET /health
GET /config/runtime
```

The frontend links to:

```txt
/docs
/openapi.json
```

For local development, these links may be absolute, for example:

```txt
http://localhost:8000/docs
http://localhost:8000/openapi.json
```

When the frontend is later served through a container or reverse proxy, these can be revisited.

## Current file structure

```txt
system/frontend/
├── package.json
├── package-lock.json
├── public/
│   ├── index.html
│   ├── css/
│   │   ├── base.css
│   │   ├── tablet.css
│   │   ├── desktop.css
│   │   ├── big.css
│   │   ├── ultrawide.css
│   │   └── print.css
│   └── js/
│       └── app.js
└── src/
    └── scss/
        ├── base.scss
        ├── tablet.scss
        ├── desktop.scss
        ├── big.scss
        ├── ultrawide.scss
        ├── print.scss
        ├── abstracts/
        ├── base/
        ├── layout/
        ├── components/
        ├── pages/
        └── device/
```

## CSS build

The frontend uses Sass and Bootstrap as dependencies.

Build command:

```bash
npm run build:css
```

Watch command:

```bash
npm run watch:css
```

The build compiles multiple device-specific CSS files:

```txt
src/scss/base.scss       -> public/css/base.css
src/scss/tablet.scss     -> public/css/tablet.css
src/scss/desktop.scss    -> public/css/desktop.css
src/scss/big.scss        -> public/css/big.css
src/scss/ultrawide.scss  -> public/css/ultrawide.css
src/scss/print.scss      -> public/css/print.css
```

## CSS loading strategy

The page loads `base.css` for all devices.

Additional CSS files are loaded only when their media query matches:

```html
<link rel="stylesheet" href="css/base.css">
<link rel="stylesheet" href="css/tablet.css" media="screen and (min-width: 768px)">
<link rel="stylesheet" href="css/desktop.css" media="screen and (min-width: 992px)">
<link rel="stylesheet" href="css/big.css" media="screen and (min-width: 1400px)">
<link rel="stylesheet" href="css/ultrawide.css" media="screen and (min-width: 2200px)">
<link rel="stylesheet" href="css/print.css" media="print">
```

This follows the same general responsive pattern as the existing `guldmand.com` styling approach.

## Layout approach

The page uses a body-level grid.

The main idea is:

- `body.page` defines the full page grid
- `header` spans the full page
- `main.page__main` inherits the grid
- page sections sit in the content column
- device CSS files adjust layout variables and grid behavior

This keeps widths controlled at the outer layout level instead of adding many unrelated max-width rules to individual components.

## Current page sections

The current shell contains:

- Header
- Hero section
- Backend status
- Runtime config
- Strategy placeholder
- Portfolio placeholder
- Decision output placeholder
- Risk output placeholder
- Audit/evidence placeholder
- Research bridge placeholder

## Notes

The frontend is intentionally more structured than the original single-file `app.css` suggestion from issue `#4`, but it is still static and minimal.

The SCSS split is considered acceptable because it supports the PoC and matches the user’s existing responsive design workflow without introducing a heavy frontend framework.
