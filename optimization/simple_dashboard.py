"""
Simplified dashboard for visualizing trading strategy optimization progress
"""
import os
import json
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import time
import threading
import logging

# Dashboard imports
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

# Setup logging
logger = logging.getLogger('simple_dashboard')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class SimpleDashboard:
    """
    Simplified dashboard for visualizing optimization progress
    """
    
    def __init__(self, checkpoint_dir='optimization_checkpoints', port=8050, stop_callback=None):
        """
        Initialize the dashboard
        
        Args:
            checkpoint_dir: Directory containing optimization checkpoints
            port: Port to run the dashboard on
            stop_callback: Function to call when the stop button is pressed
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.port = port
        self.stop_callback = stop_callback
        
        # Initialize dashboard
        self.app = dash.Dash(
            __name__,
            suppress_callback_exceptions=True,
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
        )
        
        # Set up layout
        self._setup_layout()
        
        # Register callbacks
        self._register_callbacks()
        
        logger.info(f"Initialized dashboard on port {port}")
    
    def _setup_layout(self):
        """Set up the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Trading Strategy Optimization Dashboard", style={'textAlign': 'center', 'marginBottom': '20px'}),
                html.Div([
                    html.Button(
                        'Stop Optimization', 
                        id='stop-button', 
                        style={
                            'backgroundColor': '#d9534f', 
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 20px',
                            'borderRadius': '5px',
                            'fontSize': '16px',
                            'cursor': 'pointer',
                            'margin': '10px'
                        }
                    ),
                    html.Div(id='stop-status', style={'display': 'inline-block', 'marginLeft': '10px'})
                ], style={'textAlign': 'center'}),
                html.Div([
                    html.Span("Last updated: "),
                    html.Span(id="last-update-time")
                ], style={'textAlign': 'right', 'marginTop': '10px', 'marginRight': '20px'})
            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginBottom': '20px', 'borderRadius': '5px'}),
            
            # Tabs for each strategy
            dcc.Tabs([
                dcc.Tab(label="ML Strategy", children=[
                    html.Div([
                        html.H3("Optimization Progress"),
                        dcc.Loading(
                            html.Div(id="ml-progress-stats"),
                            type="circle"
                        ),
                        html.H3("Performance Metrics"),
                        dcc.Graph(id="ml-metrics-chart"),
                        html.H3("Best Configuration"),
                        html.Pre(id="ml-best-config", style={'backgroundColor': '#f8f9fa', 'padding': '10px', 'borderRadius': '5px'})
                    ], style={'padding': '20px'})
                ]),
                dcc.Tab(label="Medallion Strategy", children=[
                    html.Div([
                        html.H3("Optimization Progress"),
                        dcc.Loading(
                            html.Div(id="medallion-progress-stats"),
                            type="circle"
                        ),
                        html.H3("Performance Metrics"),
                        dcc.Graph(id="medallion-metrics-chart"),
                        html.H3("Best Configuration"),
                        html.Pre(id="medallion-best-config", style={'backgroundColor': '#f8f9fa', 'padding': '10px', 'borderRadius': '5px'})
                    ], style={'padding': '20px'})
                ])
            ]),
            
            # Data store and refresh interval
            dcc.Store(id="optimization-data"),
            dcc.Interval(
                id="interval-component",
                interval=5000,  # 5 seconds refresh
                n_intervals=0
            )
        ])
    
    def _register_callbacks(self):
        """Register dashboard callbacks"""
        
        @self.app.callback(
            Output("last-update-time", "children"),
            [Input("interval-component", "n_intervals")]
        )
        def update_time(n):
            return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        @self.app.callback(
            Output("optimization-data", "data"),
            [Input("interval-component", "n_intervals")]
        )
        def update_data(n):
            """Load latest optimization data"""
            return self._load_latest_data()
        
        @self.app.callback(
            [Output("ml-progress-stats", "children"),
             Output("medallion-progress-stats", "children")],
            [Input("optimization-data", "data")]
        )
        def update_progress_stats(data):
            """Update progress statistics for both strategies"""
            ml_stats = html.Div("No data available for ML strategy")
            medallion_stats = html.Div("No data available for Medallion strategy")
            
            if data:
                # ML Strategy stats
                if 'ml' in data and 'total_iterations' in data['ml']:
                    ml_data = data['ml']
                    ml_stats = html.Div([
                        html.Div([
                            html.Div([
                                html.H4("Total Iterations"),
                                html.P(ml_data['total_iterations'], style={'fontSize': '24px'})
                            ], style={'flex': '1', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),
                            html.Div([
                                html.H4("Best Score"),
                                html.P(f"{ml_data.get('best_score', 0):.2f}", style={'fontSize': '24px', 'color': 'green'})
                            ], style={'flex': '1', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),
                            html.Div([
                                html.H4("Win Rate"),
                                html.P(f"{ml_data.get('best_win_rate', 0):.2f}%", style={'fontSize': '24px'})
                            ], style={'flex': '1', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),
                            html.Div([
                                html.H4("Annual Return"),
                                html.P(f"{ml_data.get('best_annual_return', 0):.2f}%", style={'fontSize': '24px'})
                            ], style={'flex': '1', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap'})
                    ])
                
                # Medallion Strategy stats
                if 'medallion' in data and 'total_iterations' in data['medallion']:
                    medallion_data = data['medallion']
                    medallion_stats = html.Div([
                        html.Div([
                            html.Div([
                                html.H4("Total Iterations"),
                                html.P(medallion_data['total_iterations'], style={'fontSize': '24px'})
                            ], style={'flex': '1', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),
                            html.Div([
                                html.H4("Best Score"),
                                html.P(f"{medallion_data.get('best_score', 0):.2f}", style={'fontSize': '24px', 'color': 'green'})
                            ], style={'flex': '1', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),
                            html.Div([
                                html.H4("Win Rate"),
                                html.P(f"{medallion_data.get('best_win_rate', 0):.2f}%", style={'fontSize': '24px'})
                            ], style={'flex': '1', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),
                            html.Div([
                                html.H4("Annual Return"),
                                html.P(f"{medallion_data.get('best_annual_return', 0):.2f}%", style={'fontSize': '24px'})
                            ], style={'flex': '1', 'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'margin': '5px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap'})
                    ])
            
            return ml_stats, medallion_stats
        
        @self.app.callback(
            [Output("ml-metrics-chart", "figure"),
             Output("medallion-metrics-chart", "figure")],
            [Input("optimization-data", "data")]
        )
        def update_metrics_charts(data):
            """Update metrics charts for both strategies"""
            empty_figure = go.Figure()
            empty_figure.update_layout(title="No data available")
            
            ml_figure = empty_figure
            medallion_figure = empty_figure
            
            if data:
                # ML Strategy chart
                if 'ml' in data and 'metrics_history' in data['ml']:
                    ml_metrics = data['ml']['metrics_history']
                    if ml_metrics:
                        df = pd.DataFrame(ml_metrics)
                        ml_figure = go.Figure()
                        
                        # Create traces for key metrics
                        if 'win_rate' in df.columns:
                            ml_figure.add_trace(go.Scatter(
                                x=df.index, y=df['win_rate'],
                                mode='lines+markers',
                                name='Win Rate (%)'
                            ))
                        
                        if 'annual_return' in df.columns:
                            ml_figure.add_trace(go.Scatter(
                                x=df.index, y=df['annual_return'],
                                mode='lines+markers',
                                name='Annual Return (%)'
                            ))
                        
                        if 'max_drawdown' in df.columns:
                            ml_figure.add_trace(go.Scatter(
                                x=df.index, y=df['max_drawdown'],
                                mode='lines+markers',
                                name='Max Drawdown (%)'
                            ))
                        
                        if 'score' in df.columns:
                            ml_figure.add_trace(go.Scatter(
                                x=df.index, y=df['score'],
                                mode='lines+markers',
                                name='Score',
                                line=dict(width=3, dash='dash')
                            ))
                        
                        ml_figure.update_layout(
                            title='ML Strategy Metrics Progress',
                            xaxis_title='Iteration',
                            yaxis_title='Value',
                            hovermode='x unified',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
                        )
                
                # Medallion Strategy chart
                if 'medallion' in data and 'metrics_history' in data['medallion']:
                    medallion_metrics = data['medallion']['metrics_history']
                    if medallion_metrics:
                        df = pd.DataFrame(medallion_metrics)
                        medallion_figure = go.Figure()
                        
                        # Create traces for key metrics
                        if 'win_rate' in df.columns:
                            medallion_figure.add_trace(go.Scatter(
                                x=df.index, y=df['win_rate'],
                                mode='lines+markers',
                                name='Win Rate (%)'
                            ))
                        
                        if 'annual_return' in df.columns:
                            medallion_figure.add_trace(go.Scatter(
                                x=df.index, y=df['annual_return'],
                                mode='lines+markers',
                                name='Annual Return (%)'
                            ))
                        
                        if 'max_drawdown' in df.columns:
                            medallion_figure.add_trace(go.Scatter(
                                x=df.index, y=df['max_drawdown'],
                                mode='lines+markers',
                                name='Max Drawdown (%)'
                            ))
                        
                        if 'score' in df.columns:
                            medallion_figure.add_trace(go.Scatter(
                                x=df.index, y=df['score'],
                                mode='lines+markers',
                                name='Score',
                                line=dict(width=3, dash='dash')
                            ))
                        
                        medallion_figure.update_layout(
                            title='Medallion Strategy Metrics Progress',
                            xaxis_title='Iteration',
                            yaxis_title='Value',
                            hovermode='x unified',
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
                        )
            
            return ml_figure, medallion_figure
        
        @self.app.callback(
            [Output("ml-best-config", "children"),
             Output("medallion-best-config", "children")],
            [Input("optimization-data", "data")]
        )
        def update_best_configs(data):
            """Update best configuration displays"""
            ml_config = "No best configuration available for ML strategy"
            medallion_config = "No best configuration available for Medallion strategy"
            
            if data:
                # ML Strategy best config
                if 'ml' in data and 'best_params' in data['ml']:
                    ml_params = data['ml']['best_params']
                    ml_metrics = data['ml'].get('best_metrics', {})
                    
                    ml_config = f"Best Parameters:\n{json.dumps(ml_params, indent=2)}\n\n"
                    ml_config += f"Best Metrics:\n{json.dumps(ml_metrics, indent=2)}"
                
                # Medallion Strategy best config
                if 'medallion' in data and 'best_params' in data['medallion']:
                    medallion_params = data['medallion']['best_params']
                    medallion_metrics = data['medallion'].get('best_metrics', {})
                    
                    medallion_config = f"Best Parameters:\n{json.dumps(medallion_params, indent=2)}\n\n"
                    medallion_config += f"Best Metrics:\n{json.dumps(medallion_metrics, indent=2)}"
            
            return ml_config, medallion_config
        
        @self.app.callback(
            Output("stop-status", "children"),
            [Input("stop-button", "n_clicks")],
            [State("stop-status", "children")]
        )
        def stop_optimization(n_clicks, current_status):
            """Handle stop button click"""
            if not n_clicks:
                return ""
            
            if callback_context.triggered:
                if self.stop_callback:
                    self.stop_callback()
                    return "Stopping optimization... (completing current batch)"
            
            return current_status
    
    def _load_latest_data(self):
        """
        Load the latest optimization data from checkpoint files
        
        Returns:
            dict: Dictionary containing optimization data
        """
        data = {
            'ml': {},
            'medallion': {}
        }
        
        try:
            # Ensure checkpoint directory exists
            self.checkpoint_dir.mkdir(exist_ok=True)
            
            # Process each strategy
            for strategy in ['ml', 'medallion']:
                strategy_dir = self.checkpoint_dir / strategy
                
                if not strategy_dir.exists():
                    continue
                
                # Find all checkpoint files for this strategy
                checkpoint_files = list(strategy_dir.glob('checkpoint_*.json'))
                metrics_files = list(strategy_dir.glob('metrics_*.csv'))
                
                # Load latest checkpoint if available
                if checkpoint_files:
                    # Get the latest checkpoint file (highest iteration number)
                    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[1]))
                    
                    with open(latest_checkpoint, 'r') as f:
                        checkpoint_data = json.load(f)
                    
                    # Extract key information
                    data[strategy]['best_params'] = checkpoint_data.get('best_params', {})
                    data[strategy]['best_metrics'] = checkpoint_data.get('best_metrics', {})
                    data[strategy]['best_score'] = checkpoint_data.get('best_score', 0)
                    data[strategy]['best_win_rate'] = checkpoint_data.get('best_metrics', {}).get('win_rate', 0)
                    data[strategy]['best_annual_return'] = checkpoint_data.get('best_metrics', {}).get('annual_return', 0)
                    
                    # Extract iteration number from file name
                    iteration = int(latest_checkpoint.stem.split('_')[1])
                    data[strategy]['total_iterations'] = iteration
                
                # Load metrics history if available
                if metrics_files:
                    # Get the latest metrics file
                    latest_metrics = max(metrics_files, key=lambda x: int(x.stem.split('_')[1]))
                    
                    # Load metrics data
                    metrics_df = pd.read_csv(latest_metrics)
                    
                    # Add to data
                    data[strategy]['metrics_history'] = metrics_df.to_dict('records')
        
        except Exception as e:
            logger.error(f"Error loading optimization data: {e}")
        
        return data
    
    def run_server(self):
        """Run the dashboard server"""
        self.app.run_server(debug=False, host='0.0.0.0', port=self.port)

if __name__ == '__main__':
    # Run the dashboard standalone
    dashboard = SimpleDashboard()
    dashboard.run_server() 