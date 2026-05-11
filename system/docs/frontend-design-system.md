# Frontend Design System

## Purpose

This document describes the lightweight frontend design system used by the StockInvestmentDSS V1.0 PoC.

The design system is intentionally small. It is not a full product design system yet. It is a practical SCSS structure for building a consistent static frontend shell that can later evolve into a Django/guldmand.com-integrated interface.

## Design principles

The current design system follows these principles:

- Mobile-first base styling
- Device-specific CSS files for larger layouts
- BEM-style class names
- Layout width controlled by the outer page grid
- Minimal component rules
- No unnecessary frontend framework
- Bootstrap allowed as a foundation, but custom DSS styles remain project-owned
- Static CSS output that can be served by Nginx later

## Naming convention

The frontend uses BEM-style naming.

Examples:

```html
<header class="header">
  <div class="header__inner">
    <a class="header__logo" href="/">StockInvestmentDSS</a>
    <nav class="header__nav">
      <a class="header__nav-link" href="#strategy">Strategy</a>
    </nav>
  </div>
</header>
```

Component examples:

```html
<section class="runtime-status">
  <article class="status-card status-card--backend">
    <h2 class="status-card__title">Backend status</h2>
    <p class="status-card__value status-card__value--pending">Checking backend...</p>
  </article>
</section>
```

Rules:

- Blocks describe standalone components or layout sections
- Elements use `__`
- Modifiers use `--`
- Avoid generic project prefixes such as `app-` unless needed later
- Keep class names readable and close to the domain

## SCSS structure

Current structure:

```txt
src/scss/
├── base.scss
├── tablet.scss
├── desktop.scss
├── big.scss
├── ultrawide.scss
├── print.scss
│
├── abstracts/
│   ├── _functions.scss
│   ├── _mixins.scss
│   ├── _colors.scss
│   └── _variables.scss
│
├── base/
│   ├── _basic.scss
│   └── _typography.scss
│
├── layout/
│   ├── _grid.scss
│   └── _header.scss
│
├── components/
│   ├── _buttons.scss
│   ├── _cards.scss
│   └── _status.scss
│
├── pages/
│   └── _home.scss
│
└── device/
    ├── _tablet.scss
    ├── _desktop.scss
    ├── _big.scss
    ├── _ultrawide.scss
    └── _print.scss
```

## Entry files

`base.scss` is the main stylesheet entry.

It imports:

- functions
- mixins
- colors
- variables
- Bootstrap
- base styles
- layout styles
- components
- page-specific styles

Device entry files are intentionally small and only import their matching device partial.

Example:

```scss
@import "device/tablet";
```

The device CSS files are loaded by media queries in `index.html`, so they should only contain overrides that are relevant for that viewport size.

## Breakpoints

Breakpoints are aligned with the CSS files loaded in `index.html`.

```scss
$breakpoints: (
  tablet: "screen and (min-width: 768px)",
  desktop: "screen and (min-width: 992px)",
  big: "screen and (min-width: 1400px)",
  ultrawide: "screen and (min-width: 2200px)",
  print: "print"
);
```

The current HTML loading strategy:

```html
<link rel="stylesheet" href="css/base.css">
<link rel="stylesheet" href="css/tablet.css" media="screen and (min-width: 768px)">
<link rel="stylesheet" href="css/desktop.css" media="screen and (min-width: 992px)">
<link rel="stylesheet" href="css/big.css" media="screen and (min-width: 1400px)">
<link rel="stylesheet" href="css/ultrawide.css" media="screen and (min-width: 2200px)">
<link rel="stylesheet" href="css/print.css" media="print">
```

## Responsive layout strategy

Base CSS applies to all devices.

Device CSS files then progressively enhance the layout:

- `tablet.css`: show navigation and move selected sections into two columns
- `desktop.css`: increase spacing and use wider grids
- `big.css`: increase page max-width and allow more columns
- `ultrawide.css`: use very wide monitors properly
- `print.css`: remove decorative styling and optimize for print

The page width is controlled through CSS custom properties:

```scss
:root {
  --page-gap: 1rem;
  --page-padding-inline: 1rem;
  --page-max-width: 1200px;
}
```

Device files update these values:

```scss
:root {
  --page-gap: 2rem;
  --page-padding-inline: 4rem;
  --page-max-width: 2200px;
}
```

## Page grid

The page grid lives in `layout/_grid.scss`.

The concept:

```scss
body.page {
  display: grid;
  grid-template-columns:
    [full-start] minmax(var(--page-padding-inline), 1fr)
    [content-start] minmax(0, var(--page-max-width))
    [content-end] minmax(var(--page-padding-inline), 1fr)
    [full-end];
  grid-template-rows: auto 1fr;
}
```

This makes it possible to keep the page full-width while still aligning content consistently.

The main content inherits the grid:

```scss
.page__main {
  display: grid;
  grid-template-columns: inherit;
}
```

Sections can then be placed in the content column:

```scss
.page__main > section {
  grid-column: content-start / content-end;
}
```

## Colors

Core project colors are defined in `abstracts/_colors.scss`.

Current color direction:

```scss
$color-black: #111111;
$color-white: #ffffff;
$color-orange: #e2802b;

$color-text: #000000;
$color-text-muted: #666666;

$color-background: #ffffff;
$color-background-muted: #f6f7f9;
$color-surface: #ffffff;

$color-border: #dee2e6;
$color-border-soft: rgba($color-black, 0.06);

$color-success: #198754;
$color-warning: #ffc107;
$color-error: #dc3545;
```

A palette-based helper can be used where useful, especially if the system later needs more controlled color variants, alpha values, or contrast-aware functions.

## Functions

Current function direction:

```scss
@function rem($px, $base: 16) {
  @return calc($px / $base) * 1rem;
}
```

Additional useful functions from the existing portfolio style system may be used:

```scss
@function color-hsl($color) { ... }
@function color-hsla($color) { ... }
@function color-custom-hsl-with-alpha($color, $alpha) { ... }
@function color($base, $shade: base, $alpha: 1) { ... }
@function color-hex($color, $tone: "base", $a: 1) { ... }
@function calculateRem($size) { ... }
```

These are useful when the design system grows beyond the current PoC shell.

## Mixins

Useful mixins:

```scss
@mixin focus-ring {
  outline: 2px solid rgba($color-orange, 0.7);
  outline-offset: 3px;
}
```

```scss
@mixin respond-to($breakpoint) {
  $raw-query: map-get($breakpoints, $breakpoint);

  @if $raw-query {
    $query: if(
      type-of($raw-query) == "string",
      unquote($raw-query),
      inspect($raw-query)
    );

    @media #{$query} {
      @content;
    }
  } @else {
    @error "No value found for `#{$breakpoint}`. Please make sure it is defined in `$breakpoints` map.";
  }
}
```

For the current setup, most large responsive changes live in the device CSS files. The `respond-to()` mixin is still useful for smaller local exceptions inside base/layout/component files.

## Components

Current component groups:

```txt
components/_buttons.scss
components/_cards.scss
components/_status.scss
```

### Buttons

Button classes:

```html
<a class="button button--primary">Open API docs</a>
<a class="button button--secondary">OpenAPI JSON</a>
```

Buttons should remain simple, accessible links or buttons.

### Cards

Card types:

```html
<article class="status-card status-card--backend">...</article>
<article class="dss-card dss-card--strategy">...</article>
```

Card styling should stay in `components/_cards.scss`.

Grid placement should stay in layout or device files.

### Status

Backend health status uses modifier classes:

```html
<p class="status-card__value status-card__value--pending">Checking backend...</p>
```

JavaScript may add status classes such as:

```txt
status--ok
status--error
```

These should remain visual-only. The text content should also change so the status is understandable without color.

## Typography

Typography is currently simple and based on the Typekit font loaded in `index.html`:

```html
<link rel="stylesheet" href="https://use.typekit.net/nco1sey.css">
```

Fallback font stack:

```scss
$font-family-sans-serif: "myriad-pro", "Helvetica Neue", Arial, sans-serif;
```

Typography rules should avoid excessive letter-spacing unless it is specifically needed for a design effect.

## Bootstrap usage

Bootstrap is imported as a Sass dependency.

Current approach:

- Bootstrap may provide base normalization and useful utilities
- Project styling should remain custom and BEM-based
- Avoid building the PoC UI around Bootstrap-specific markup
- Avoid vendoring external dashboard repositories

Bootstrap is useful, but the DSS frontend should remain lightweight and project-owned.

## Print styling

Print styling belongs in:

```txt
device/_print.scss
```

Print CSS should:

- Hide navigation and interactive buttons
- Remove shadows
- Use black text on white background
- Avoid card splitting where possible
- Optionally show link URLs after links

## What belongs where

Use this rule:

```txt
abstracts/  -> tokens, functions, mixins, variables
base/       -> global element styles
layout/     -> page grid, header, footer, navigation layout
components/ -> reusable components like buttons, cards, status
pages/      -> page-specific sections such as hero/home page
device/     -> responsive overrides for loaded CSS files
```

Avoid adding layout width rules inside random components unless the component itself truly owns the width.

## Current status

The current design system is good enough for V1.0 PoC.

It supports:

- local frontend shell
- backend status display
- runtime status display
- responsive layout
- ultrawide layout
- future container serving
- future Django/guldmand.com integration

Further polish should wait until after:

- `#163 Create frontend container and add it to Docker Compose`
- `#39 Verify local app: front page and login`
