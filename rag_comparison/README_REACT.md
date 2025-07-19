# RAG Comparison Tool - React Frontend

This is a React-based frontend for the RAG Comparison Tool, replacing the previous HTML/JavaScript implementation for better performance and maintainability.

## Features

- **Modern React Architecture**: Built with React 18 and hooks
- **Real-time Comparisons**: Compare Traditional RAG vs Graph-Enhanced RAG
- **Interactive Charts**: Visualize performance metrics with Chart.js
- **Responsive Design**: Works on desktop and mobile devices
- **API Integration**: Connects to Flask backend for data processing

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn

### Installation

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

3. Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

### Building for Production

1. Build the app:
   ```bash
   npm run build
   ```

2. The build files will be in the `build/` directory.

3. Start the Flask server to serve the built React app:
   ```bash
   python server.py
   ```

## Component Structure

- **App.js**: Main application component with state management
- **Header.js**: Application header with title and description
- **QuerySection.js**: Query input and preset query buttons
- **ComparisonStats.js**: Summary statistics display
- **ChartContainer.js**: Chart.js integration for visualizations
- **ComparisonContainer.js**: Side-by-side RAG comparison results
- **LoadingIndicator.js**: Loading spinner component

## API Integration

The React app connects to the Flask backend using axios for:
- `/api/compare` - Single comparison endpoint
- `/api/batch_compare` - Batch comparison endpoint
- `/api/system_info` - System information endpoint

## Development vs Production

- **Development**: Run `npm start` for hot reloading at localhost:3000
- **Production**: Run `npm run build` then serve via Flask at localhost:5000

## Styling

Uses CSS modules with:
- Modern gradient backgrounds
- Glassmorphism effects
- Responsive grid layouts
- Smooth animations and transitions

## Performance Improvements

- React's virtual DOM for efficient updates
- Component-based architecture for better organization
- Optimized re-renders with proper state management
- Lazy loading and code splitting ready 