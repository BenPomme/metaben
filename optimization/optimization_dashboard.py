"""
Real-time dashboard for visualizing trading strategy optimization progress
Uses Dash and Plotly for interactive visualization
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
import queue

# Dashboard imports
import dash
from dash import dcc, html, callback, Output, Input, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# Setup logging
logger = logging.getLogger('optimization_dashboard')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class OptimizationDashboard:
    """
    Interactive dashboard for visualizing optimization progress in real-time
    """
    
    def __init__(self, config_path='config/optimization_config.json', checkpoint_dir='optimization_checkpoints', message_queue=None, stop_callback=None):
        """
        Initialize the dashboard
        
        Args:
            config_path: Path to optimization configuration file
            checkpoint_dir: Directory containing optimization checkpoints
            message_queue: Queue for receiving messages from the controller
            stop_callback: Function to call when the stop button is pressed
        """
        self.config_path = config_path
        self.checkpoint_dir = Path(checkpoint_dir)
        self.message_queue = message_queue or queue.Queue()
        self.stop_callback = stop_callback
        self.optimization_running = True
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Dashboard settings
        self.host = self.config.get('dashboard', {}).get('host', '0.0.0.0')
        self.port = self.config.get('dashboard', {}).get('port', 8050)
        self.debug = self.config.get('dashboard', {}).get('debug', False)
        self.update_interval = self.config.get('dashboard', {}).get('update_interval_ms', 5000)
        self.max_points = self.config.get('dashboard', {}).get('max_points_display', 1000)
        
        # Optimization criteria
        self.criteria = self.config.get('criteria', {})
        
        # Data for active optimizations
        self.active_optimizations = {}
        
        # Initialize the Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True,
            meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1"}
            ]
        )
        
        # Set up app layout
        self._setup_layout()
        
        # Register callbacks
        self._register_callbacks()
        
        logger.info("Initialized optimization dashboard")
    
    def _setup_layout(self):
        """Define the dashboard layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Trading Strategy Optimization Dashboard", className="text-center my-4"),
                    html.Hr(),
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Active Optimizations"),
                    dcc.Loading(
                        id="loading-active-optimizations",
                        type="circle",
                        children=[html.Div(id="active-optimizations-content")]
                    ),
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Performance Metrics"),
                    dbc.Tabs([
                        dbc.Tab(label="ML Strategy", tab_id="tab-ml"),
                        dbc.Tab(label="Medallion Strategy", tab_id="tab-medallion"),
                    ], id="strategy-tabs"),
                    dcc.Loading(
                        id="loading-metrics",
                        type="circle",
                        children=[html.Div(id="metrics-content")]
                    ),
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Best Configuration"),
                    dcc.Loading(
                        id="loading-best-config",
                        type="circle",
                        children=[html.Div(id="best-config-content")]
                    ),
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Parameter Importance"),
                    dcc.Loading(
                        id="loading-param-importance",
                        type="circle",
                        children=[html.Div(id="param-importance-content")]
                    ),
                ], width=12)
            ]),
            
            # Hidden div for storing data
            html.Div(id='optimization-data-store', style={'display': 'none'}),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval, # in milliseconds
                n_intervals=0
            ),
            
            # Footer
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P("Last updated: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                           id="last-update-time", className="text-muted text-end"),
                ])
            ]),
            
            # Stop button
            dbc.Row([
                dbc.Col([
                    html.Button('Stop Optimization', id='stop-button', 
                                style={'background-color': '#d9534f', 'color': 'white', 
                                       'border': 'none', 'padding': '10px 20px',
                                       'border-radius': '5px', 'cursor': 'pointer'}),
                    html.Span(id='stop-status', style={'margin-left': '10px'})
                ], width=12)
            ]),
        ], fluid=True)
    
    def _register_callbacks(self):
        """Register dashboard callbacks"""
        
        @self.app.callback(
            Output("last-update-time", "children"),
            Input("interval-component", "n_intervals")
        )
        def update_time(n):
            return "Last updated: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        @self.app.callback(
            Output("optimization-data-store", "children"),
            Input("interval-component", "n_intervals")
        )
        def update_data_store(n):
            """Load and process the latest optimization data"""
            try:
                optimizations = self._load_latest_data()
                return json.dumps(optimizations)
            except Exception as e:
                logger.error(f"Error updating data store: {e}")
                return json.dumps({})
        
        @self.app.callback(
            Output("active-optimizations-content", "children"),
            Input("optimization-data-store", "children")
        )
        def update_active_optimizations(json_data):
            """Update the active optimizations display"""
            if not json_data:
                return html.Div("No active optimizations found.")
            
            try:
                optimizations = json.loads(json_data)
                
                if not optimizations:
                    return html.Div("No active optimizations found.")
                
                cards = []
                
                for strategy_type, data in optimizations.items():
                    symbol = data.get('symbol', 'Unknown')
                    timeframe = data.get('timeframe', 'Unknown')
                    iterations = data.get('iterations', 0)
                    best_score = data.get('best_score', 0)
                    criteria_met = data.get('criteria_met', 0)
                    
                    card = dbc.Card([
                        dbc.CardHeader(f"{strategy_type.upper()} Strategy - {symbol} {timeframe}"),
                        dbc.CardBody([
                            html.H5(f"Iterations: {iterations}", className="card-title"),
                            html.P([
                                html.Span("Best Score: "),
                                html.Span(f"{best_score:.2f}", 
                                          style={'color': 'green' if best_score > 50 else 'orange'})
                            ]),
                            html.P([
                                html.Span("Criteria Met: "),
                                html.Span(f"{criteria_met}", 
                                          style={'color': 'green' if criteria_met > 0 else 'red'})
                            ]),
                            dbc.Progress(value=min(100, iterations/10), 
                                         color="success", style={"height": "20px"}),
                        ])
                    ], className="mb-3")
                    
                    cards.append(card)
                
                return html.Div(cards)
                
            except Exception as e:
                logger.error(f"Error updating active optimizations: {e}")
                return html.Div(f"Error loading optimization data: {str(e)}")
        
        @self.app.callback(
            Output("metrics-content", "children"),
            [Input("optimization-data-store", "children"),
             Input("strategy-tabs", "active_tab")]
        )
        def update_metrics_charts(json_data, active_tab):
            """Update the metrics visualization"""
            if not json_data or not active_tab:
                return html.Div("No data available.")
            
            try:
                strategy_type = active_tab.split('-')[1]  # Extract strategy type from tab ID
                
                optimizations = json.loads(json_data)
                
                if not optimizations or strategy_type not in optimizations:
                    return html.Div(f"No data available for {strategy_type} strategy.")
                
                data = optimizations[strategy_type]
                metrics_data = data.get('metrics_df', [])
                
                if not metrics_data:
                    return html.Div(f"No metrics data available for {strategy_type} strategy.")
                
                # Convert metrics data to DataFrame
                df = pd.DataFrame(metrics_data)
                
                # Create charts for key metrics
                charts = []
                
                key_metrics = ['win_rate', 'annual_return', 'max_drawdown', 
                              'profit_factor', 'sharpe_ratio', 'score']
                
                for metric in key_metrics:
                    if metric in df.columns:
                        fig = go.Figure()
                        
                        # Add scatter plot for the metric
                        fig.add_trace(go.Scatter(
                            x=df['iteration'],
                            y=df[metric],
                            mode='markers+lines',
                            name=metric,
                            marker=dict(size=8),
                            line=dict(width=2)
                        ))
                        
                        # Add moving average if we have enough points
                        if len(df) > 5:
                            window = min(10, len(df))
                            df[f"{metric}_ma"] = df[metric].rolling(window=window).mean()
                            fig.add_trace(go.Scatter(
                                x=df['iteration'],
                                y=df[f"{metric}_ma"],
                                mode='lines',
                                name=f"{window}-iter MA",
                                line=dict(width=3, dash='dash', color='red')
                            ))
                        
                        # Add criteria threshold line if applicable
                        criteria_value = None
                        if metric == 'win_rate':
                            criteria_value = self.criteria.get('min_win_rate')
                        elif metric == 'annual_return':
                            criteria_value = self.criteria.get('min_annual_return')
                        elif metric == 'max_drawdown':
                            criteria_value = self.criteria.get('max_drawdown')
                        elif metric == 'profit_factor':
                            criteria_value = self.criteria.get('min_profit_factor')
                        elif metric == 'sharpe_ratio':
                            criteria_value = self.criteria.get('min_sharpe_ratio')
                        
                        if criteria_value is not None:
                            fig.add_shape(
                                type="line",
                                x0=df['iteration'].min(),
                                x1=df['iteration'].max(),
                                y0=criteria_value,
                                y1=criteria_value,
                                line=dict(color="green", width=2, dash="dash"),
                            )
                            
                            fig.add_annotation(
                                x=df['iteration'].max(),
                                y=criteria_value,
                                text=f"Target: {criteria_value}",
                                showarrow=False,
                                yshift=10,
                                bgcolor="rgba(255, 255, 255, 0.8)"
                            )
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{metric.replace('_', ' ').title()} vs Iteration",
                            xaxis_title="Iteration",
                            yaxis_title=metric.replace('_', ' ').title(),
                            hovermode="x unified",
                            height=400,
                            margin=dict(l=20, r=20, t=40, b=20),
                        )
                        
                        charts.append(dcc.Graph(figure=fig, className="mb-4"))
                
                # Layout charts in rows of 2
                rows = []
                for i in range(0, len(charts), 2):
                    cols = []
                    for j in range(2):
                        if i + j < len(charts):
                            cols.append(dbc.Col(charts[i + j], width=6))
                    
                    rows.append(dbc.Row(cols, className="mb-4"))
                
                return html.Div(rows)
                
            except Exception as e:
                logger.error(f"Error updating metrics charts: {e}")
                return html.Div(f"Error visualizing metrics data: {str(e)}")
        
        @self.app.callback(
            Output("best-config-content", "children"),
            [Input("optimization-data-store", "children"),
             Input("strategy-tabs", "active_tab")]
        )
        def update_best_config(json_data, active_tab):
            """Update the best configuration display"""
            if not json_data or not active_tab:
                return html.Div("No data available.")
            
            try:
                strategy_type = active_tab.split('-')[1]  # Extract strategy type from tab ID
                
                optimizations = json.loads(json_data)
                
                if not optimizations or strategy_type not in optimizations:
                    return html.Div(f"No data available for {strategy_type} strategy.")
                
                data = optimizations[strategy_type]
                best_config = data.get('best_config', {})
                
                if not best_config:
                    return html.Div(f"No best configuration available for {strategy_type} strategy.")
                
                params = best_config.get('params', {})
                metrics = best_config.get('metrics', {})
                
                param_items = []
                for k, v in params.items():
                    param_items.append(html.Tr([
                        html.Td(k.replace('_', ' ').title()),
                        html.Td(str(v))
                    ]))
                
                metric_items = []
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        formatted_value = f"{v:.4f}" if isinstance(v, float) else str(v)
                    else:
                        formatted_value = str(v)
                    
                    metric_items.append(html.Tr([
                        html.Td(k.replace('_', ' ').title()),
                        html.Td(formatted_value)
                    ]))
                
                # Create card with parameter and metric tables
                card = dbc.Card([
                    dbc.CardHeader(f"Best Configuration for {strategy_type.upper()} Strategy"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Parameters"),
                                html.Div([
                                    dbc.Table([
                                        html.Thead(html.Tr([
                                            html.Th("Parameter"),
                                            html.Th("Value")
                                        ])),
                                        html.Tbody(param_items)
                                    ], striped=True, bordered=True, hover=True, responsive=True)
                                ], style={"max-height": "500px", "overflow-y": "auto"})
                            ], width=6),
                            dbc.Col([
                                html.H5("Metrics"),
                                dbc.Table([
                                    html.Thead(html.Tr([
                                        html.Th("Metric"),
                                        html.Th("Value")
                                    ])),
                                    html.Tbody(metric_items)
                                ], striped=True, bordered=True, hover=True, responsive=True)
                            ], width=6)
                        ])
                    ])
                ], className="mb-3")
                
                return card
                
            except Exception as e:
                logger.error(f"Error updating best config: {e}")
                return html.Div(f"Error loading best configuration: {str(e)}")
        
        @self.app.callback(
            Output("param-importance-content", "children"),
            [Input("optimization-data-store", "children"),
             Input("strategy-tabs", "active_tab")]
        )
        def update_param_importance(json_data, active_tab):
            """Update the parameter importance visualization"""
            if not json_data or not active_tab:
                return html.Div("No data available.")
            
            try:
                strategy_type = active_tab.split('-')[1]  # Extract strategy type from tab ID
                
                optimizations = json.loads(json_data)
                
                if not optimizations or strategy_type not in optimizations:
                    return html.Div(f"No data available for {strategy_type} strategy.")
                
                data = optimizations[strategy_type]
                metrics_data = data.get('metrics_df', [])
                
                if not metrics_data or len(metrics_data) < 10:
                    return html.Div(f"Not enough data to calculate parameter importance yet (need at least 10 iterations).")
                
                # Convert metrics data to DataFrame
                df = pd.DataFrame(metrics_data)
                
                # Extract parameter columns (prefixed with 'param_')
                param_cols = [col for col in df.columns if col.startswith('param_')]
                
                if not param_cols:
                    return html.Div("No parameter data found.")
                
                # Calculate correlation with score
                correlations = []
                for col in param_cols:
                    try:
                        # Only numeric parameters can be correlated
                        if pd.api.types.is_numeric_dtype(df[col]):
                            corr = df[col].corr(df['score'])
                            if not pd.isna(corr):
                                correlations.append({
                                    'parameter': col.replace('param_', '').replace('_', ' ').title(),
                                    'correlation': corr,
                                    'abs_correlation': abs(corr)
                                })
                    except Exception:
                        pass
                
                if not correlations:
                    return html.Div("Could not calculate parameter importance (requires numeric parameters).")
                
                # Sort by absolute correlation
                correlations_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
                
                # Take top 15 parameters
                top_params = correlations_df.head(15)
                
                # Create bar chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    y=top_params['parameter'],
                    x=top_params['correlation'],
                    orientation='h',
                    marker_color=top_params['correlation'].apply(lambda x: 'green' if x > 0 else 'red')
                ))
                
                fig.update_layout(
                    title="Parameter Importance (Correlation with Score)",
                    xaxis_title="Correlation Coefficient",
                    yaxis_title="Parameter",
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                return dcc.Graph(figure=fig)
                
            except Exception as e:
                logger.error(f"Error updating parameter importance: {e}")
                return html.Div(f"Error calculating parameter importance: {str(e)}")
        
        @self.app.callback(
            Output('stop-status', 'children'),
            [Input('stop-button', 'n_clicks')],
            [State('stop-status', 'children')]
        )
        def stop_optimization(n_clicks, current_status):
            if not n_clicks:
                return ""
            
            if callback_context.triggered:
                if self.stop_callback:
                    self.stop_callback()
                    self.optimization_running = False
                    return "Stopping optimization..."
            
            return current_status
    
    def _load_latest_data(self):
        """
        Load the latest optimization data from checkpoint files
        
        Returns:
            dict: Dictionary of optimization data by strategy type
        """
        optimizations = {}
        
        # Create checkpoint dir if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Scan the checkpoint directory for JSON files
        for file_path in self.checkpoint_dir.glob('*_best_config.json'):
            try:
                # Extract strategy type and symbol from filename
                file_name = file_path.stem
                parts = file_name.split('_')
                if len(parts) >= 3:
                    strategy_type = parts[0]
                    symbol = parts[1]
                    timeframe = parts[2]
                    
                    # Load the best configuration
                    with open(file_path, 'r') as f:
                        best_config = json.load(f)
                    
                    # Find the latest metrics file for this strategy/symbol/timeframe
                    metrics_files = list(self.checkpoint_dir.glob(f'{strategy_type}_{symbol}_{timeframe}_metrics_*.csv'))
                    if metrics_files:
                        # Sort by modification time (latest first)
                        latest_metrics_file = sorted(metrics_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                        
                        # Load the metrics data
                        metrics_df = pd.read_csv(latest_metrics_file)
                        
                        # Count iterations and criteria met
                        iterations = len(metrics_df)
                        criteria_met = 0
                        
                        # Calculate criteria met count if we have the criteria columns
                        if all(col in metrics_df.columns for col in ['win_rate', 'annual_return', 'max_drawdown']):
                            criteria_met = len(metrics_df[
                                (metrics_df['win_rate'] >= self.criteria.get('min_win_rate', 51.0)) &
                                (metrics_df['annual_return'] >= self.criteria.get('min_annual_return', 25.0)) &
                                (metrics_df['max_drawdown'] <= self.criteria.get('max_drawdown', 10.0))
                            ])
                        
                        # Prepare data for dashboard
                        optimizations[strategy_type] = {
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'best_config': best_config,
                            'metrics_df': metrics_df.to_dict('records'),
                            'iterations': iterations,
                            'best_score': best_config.get('score', 0),
                            'criteria_met': criteria_met
                        }
            except Exception as e:
                logger.error(f"Error loading checkpoint file {file_path}: {e}")
        
        return optimizations
    
    def run_server(self, in_background=False):
        """
        Run the dashboard server
        
        Args:
            in_background: Whether to run the server in a background thread
        """
        if in_background:
            # Run in background thread
            thread = threading.Thread(target=self._run_server)
            thread.daemon = True
            thread.start()
            logger.info(f"Dashboard started in background at http://{self.host}:{self.port}")
            return thread
        else:
            # Run in current thread (blocking)
            logger.info(f"Starting dashboard at http://{self.host}:{self.port}")
            self._run_server()
    
    def _run_server(self):
        """Internal method to run the server"""
        self.app.run_server(
            host=self.host,
            port=self.port,
            debug=self.debug
        )

# If run directly, start the dashboard
if __name__ == "__main__":
    dashboard = OptimizationDashboard()
    dashboard.run_server() 