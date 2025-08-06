#!/usr/bin/env python3
"""
Markov Decision Process Reverse Engineering for Polymarket Trading
Uses deep learning to model and predict user decision-making patterns
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class TradingStateEncoder:
    """Encodes trading states and market conditions into feature vectors"""
    
    def __init__(self):
        self.price_scaler = StandardScaler()
        self.volume_scaler = StandardScaler()
        self.time_scaler = StandardScaler()
        self.outcome_encoder = LabelEncoder()
        self.market_encoder = LabelEncoder()
        self.fitted = False
        
    def extract_market_features(self, activities: List[Dict]) -> pd.DataFrame:
        """Extract market-level features from activities"""
        features = []
        
        for i, activity in enumerate(activities):
            # Basic activity features
            price = float(activity.get('price', 0.5))
            volume = float(activity.get('usdcSize', 0))
            shares = float(activity.get('size', 0))
            
            # Time features
            timestamp = activity.get('timestamp', '')
            if isinstance(timestamp, str):
                try:
                    dt = pd.to_datetime(timestamp)
                    hour_of_day = dt.hour
                    day_of_week = dt.weekday()
                    unix_time = dt.timestamp()
                except:
                    hour_of_day = 12
                    day_of_week = 0
                    unix_time = 0
            else:
                hour_of_day = 12
                day_of_week = 0
                unix_time = float(timestamp) if timestamp else 0
            
            # Market context (look at recent activities)
            recent_volume = 0
            recent_price_trend = 0
            volatility = 0
            
            if i > 0:
                # Calculate recent market activity (last 5 trades)
                lookback = min(5, i)
                recent_activities = activities[max(0, i-lookback):i]
                
                if recent_activities:
                    recent_prices = [float(a.get('price', 0.5)) for a in recent_activities]
                    recent_volumes = [float(a.get('usdcSize', 0)) for a in recent_activities]
                    
                    recent_volume = sum(recent_volumes)
                    if len(recent_prices) > 1:
                        recent_price_trend = recent_prices[-1] - recent_prices[0]
                        volatility = np.std(recent_prices)
            
            # Position context
            current_position = self._calculate_position_at_time(activities[:i+1])
            position_value = current_position.get('total_value', 0)
            position_pnl = current_position.get('total_pnl', 0)
            
            feature_dict = {
                'price': price,
                'volume': volume,
                'shares': shares,
                'hour_of_day': hour_of_day,
                'day_of_week': day_of_week,
                'unix_time': unix_time,
                'recent_volume': recent_volume,
                'recent_price_trend': recent_price_trend,
                'volatility': volatility,
                'position_value': position_value,
                'position_pnl': position_pnl,
                'outcome': activity.get('outcome', 'unknown'),
                'market': activity.get('title', 'unknown'),
                'side': activity.get('side', 'unknown'),
                'activity_type': activity.get('type', 'unknown')
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _calculate_position_at_time(self, activities: List[Dict]) -> Dict:
        """Calculate position value at a specific point in time"""
        positions = defaultdict(lambda: {'shares': 0, 'cost': 0})
        
        for activity in activities:
            outcome = activity.get('outcome', '').lower()
            side = activity.get('side', '').upper()
            shares = float(activity.get('size', 0))
            cost = float(activity.get('usdcSize', 0))
            
            if side == 'BUY':
                positions[outcome]['shares'] += shares
                positions[outcome]['cost'] += cost
            elif side == 'SELL':
                positions[outcome]['shares'] -= shares
                positions[outcome]['cost'] -= cost
        
        total_value = sum(pos['shares'] * 0.5 for pos in positions.values())  # Assume 0.5 default price
        total_cost = sum(pos['cost'] for pos in positions.values())
        total_pnl = total_value - total_cost
        
        return {
            'total_value': total_value,
            'total_cost': total_cost,
            'total_pnl': total_pnl,
            'positions': dict(positions)
        }
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit encoders and transform data"""
        # Handle categorical variables
        df['outcome_encoded'] = self.outcome_encoder.fit_transform(df['outcome'].astype(str))
        df['market_encoded'] = self.market_encoder.fit_transform(df['market'].astype(str))
        
        # Handle side encoding
        side_mapping = {'BUY': 1, 'SELL': -1, 'unknown': 0}
        df['side_encoded'] = df['side'].map(side_mapping).fillna(0)
        
        # Handle activity type encoding
        type_mapping = {'TRADE': 1, 'MERGE': 0, 'REDEEM': -1, 'SPLIT': 0.5}
        df['type_encoded'] = df['activity_type'].map(type_mapping).fillna(0)
        
        # Scale numerical features
        numerical_features = ['price', 'volume', 'shares', 'recent_volume', 
                             'recent_price_trend', 'volatility', 'position_value', 'position_pnl']
        
        for feature in numerical_features:
            if feature in df.columns:
                scaler = getattr(self, f'{feature}_scaler', StandardScaler())
                df[f'{feature}_scaled'] = scaler.fit_transform(df[[feature]])
                setattr(self, f'{feature}_scaler', scaler)
        
        # Time features (cyclical encoding)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Select final features
        feature_columns = [
            'price_scaled', 'volume_scaled', 'shares_scaled',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'recent_volume_scaled', 'recent_price_trend_scaled', 'volatility_scaled',
            'position_value_scaled', 'position_pnl_scaled',
            'outcome_encoded', 'market_encoded', 'side_encoded', 'type_encoded'
        ]
        
        # Ensure all columns exist
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        self.fitted = True
        self.feature_columns = feature_columns
        
        return df[feature_columns].values
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted encoders"""
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        # Apply same transformations as fit_transform
        df = df.copy()
        
        # Handle new categories in categorical variables
        for outcome in df['outcome']:
            if outcome not in self.outcome_encoder.classes_:
                # Add new class
                self.outcome_encoder.classes_ = np.append(self.outcome_encoder.classes_, outcome)
        
        df['outcome_encoded'] = self.outcome_encoder.transform(df['outcome'].astype(str))
        
        # Similar handling for market encoder
        for market in df['market']:
            if market not in self.market_encoder.classes_:
                self.market_encoder.classes_ = np.append(self.market_encoder.classes_, market)
        
        df['market_encoded'] = self.market_encoder.transform(df['market'].astype(str))
        
        # Apply other transformations...
        side_mapping = {'BUY': 1, 'SELL': -1, 'unknown': 0}
        df['side_encoded'] = df['side'].map(side_mapping).fillna(0)
        
        type_mapping = {'TRADE': 1, 'MERGE': 0, 'REDEEM': -1, 'SPLIT': 0.5}
        df['type_encoded'] = df['activity_type'].map(type_mapping).fillna(0)
        
        # Scale numerical features using fitted scalers
        numerical_features = ['price', 'volume', 'shares', 'recent_volume', 
                             'recent_price_trend', 'volatility', 'position_value', 'position_pnl']
        
        for feature in numerical_features:
            if feature in df.columns:
                scaler = getattr(self, f'{feature}_scaler')
                df[f'{feature}_scaled'] = scaler.transform(df[[feature]])
        
        # Time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Ensure all columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        return df[self.feature_columns].values


class MarkovDecisionDataset(Dataset):
    """Dataset for training Markov Decision Process models"""
    
    def __init__(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, 
                 sequence_length: int = 10):
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)
        self.rewards = torch.FloatTensor(rewards)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.states) - self.sequence_length
    
    def __getitem__(self, idx):
        # Return sequence of states and the action taken at the end
        state_sequence = self.states[idx:idx + self.sequence_length]
        next_action = self.actions[idx + self.sequence_length - 1]
        reward = self.rewards[idx + self.sequence_length - 1]
        
        return {
            'states': state_sequence,
            'action': next_action,
            'reward': reward
        }


class LSTMDecisionModel(nn.Module):
    """LSTM-based model for predicting trading decisions"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, 
                 num_actions: int = 3, dropout: float = 0.2):
        super(LSTMDecisionModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)
        
        # Decision layers
        self.decision_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # Value estimation (for reward prediction)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        # Reshape for attention: (seq_len, batch, hidden_dim)
        lstm_out_transposed = lstm_out.transpose(0, 1)
        attended_out, attention_weights = self.attention(
            lstm_out_transposed, lstm_out_transposed, lstm_out_transposed
        )
        
        # Use the last timestep output
        final_hidden = attended_out[-1]  # (batch, hidden_dim)
        
        # Predict action probabilities
        action_logits = self.decision_layers(final_hidden)
        
        # Predict state value
        state_value = self.value_head(final_hidden)
        
        return {
            'action_logits': action_logits,
            'state_value': state_value,
            'attention_weights': attention_weights
        }


class TransformerDecisionModel(nn.Module):
    """Transformer-based model for predicting trading decisions"""
    
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 4, num_actions: int = 3, dropout: float = 0.1):
        super(TransformerDecisionModel, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer forward pass
        transformer_out = self.transformer(x)
        
        # Use the last timestep for prediction
        final_state = transformer_out[:, -1, :]  # (batch, d_model)
        
        # Predict actions and values
        action_logits = self.action_head(final_state)
        state_value = self.value_head(final_state)
        
        return {
            'action_logits': action_logits,
            'state_value': state_value
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class MarkovDecisionReverseEngineer:
    """Main class for reverse engineering trading decisions using deep learning"""
    
    def __init__(self, model_type: str = 'lstm', sequence_length: int = 10):
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.state_encoder = TradingStateEncoder()
        self.action_encoder = LabelEncoder()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
    def prepare_data(self, activities: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data from activities"""
        print("Extracting features from activities...")
        
        # Extract features
        df = self.state_encoder.extract_market_features(activities)
        states = self.state_encoder.fit_transform(df)
        
        # Create action labels
        actions = []
        rewards = []
        
        for i, activity in enumerate(activities):
            # Define action space: 0=Hold, 1=Buy, 2=Sell
            side = activity.get('side', '').upper()
            if side == 'BUY':
                action = 1
            elif side == 'SELL':
                action = 2
            else:
                action = 0  # Hold/Other
            
            actions.append(action)
            
            # Calculate reward (simplified P&L)
            price = float(activity.get('price', 0.5))
            volume = float(activity.get('usdcSize', 0))
            
            # Simple reward: positive for profitable trades, negative for losses
            # This is a simplified version - in practice, you'd calculate actual P&L
            if side == 'BUY':
                reward = (1.0 - price) * volume  # Reward for buying low
            elif side == 'SELL':
                reward = price * volume  # Reward for selling high
            else:
                reward = 0.0
            
            rewards.append(reward)
        
        actions = np.array(actions)
        rewards = np.array(rewards)
        
        print(f"Prepared {len(states)} state vectors with {states.shape[1]} features")
        print(f"Action distribution: Hold={np.sum(actions==0)}, Buy={np.sum(actions==1)}, Sell={np.sum(actions==2)}")
        
        return states, actions, rewards
    
    def create_model(self, input_dim: int, num_actions: int = 3):
        """Create the neural network model"""
        if self.model_type == 'lstm':
            model = LSTMDecisionModel(input_dim, num_actions=num_actions)
        elif self.model_type == 'transformer':
            model = TransformerDecisionModel(input_dim, num_actions=num_actions)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def train(self, activities: List[Dict], epochs: int = 100, batch_size: int = 32, 
              learning_rate: float = 0.001, validation_split: float = 0.2):
        """Train the model on activity data"""
        
        # Prepare data
        states, actions, rewards = self.prepare_data(activities)
        
        # Split data
        train_states, val_states, train_actions, val_actions, train_rewards, val_rewards = \
            train_test_split(states, actions, rewards, test_size=validation_split, random_state=42)
        
        # Create datasets
        train_dataset = MarkovDecisionDataset(train_states, train_actions, train_rewards, 
                                            self.sequence_length)
        val_dataset = MarkovDecisionDataset(val_states, val_actions, val_rewards, 
                                          self.sequence_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        input_dim = states.shape[1]
        num_actions = len(np.unique(actions))
        self.model = self.create_model(input_dim, num_actions)
        
        # Loss functions and optimizer
        action_criterion = nn.CrossEntropyLoss()
        value_criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        print(f"Training {self.model_type} model for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                states_batch = batch['states'].to(self.device)
                actions_batch = batch['action'].to(self.device)
                rewards_batch = batch['reward'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(states_batch)
                action_logits = outputs['action_logits']
                state_values = outputs['state_value'].squeeze()
                
                # Combined loss
                action_loss = action_criterion(action_logits, actions_batch)
                value_loss = value_criterion(state_values, rewards_batch)
                total_loss = action_loss + 0.5 * value_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(action_logits.data, 1)
                train_total += actions_batch.size(0)
                train_correct += (predicted == actions_batch).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    states_batch = batch['states'].to(self.device)
                    actions_batch = batch['action'].to(self.device)
                    rewards_batch = batch['reward'].to(self.device)
                    
                    outputs = self.model(states_batch)
                    action_logits = outputs['action_logits']
                    state_values = outputs['state_value'].squeeze()
                    
                    action_loss = action_criterion(action_logits, actions_batch)
                    value_loss = value_criterion(state_values, rewards_batch)
                    total_loss = action_loss + 0.5 * value_loss
                    
                    val_loss += total_loss.item()
                    
                    _, predicted = torch.max(action_logits.data, 1)
                    val_total += actions_batch.size(0)
                    val_correct += (predicted == actions_batch).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            
            # Store history
            self.training_history['loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['accuracy'].append(train_accuracy)
            self.training_history['val_accuracy'].append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f'Epoch [{epoch+1}/{epochs}]')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
                print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        print("Training completed!")
        
    def predict_next_action(self, recent_activities: List[Dict]) -> Dict:
        """Predict the next action given recent activities"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare input data
        df = self.state_encoder.extract_market_features(recent_activities)
        states = self.state_encoder.transform(df)
        
        # Take the last sequence_length states
        if len(states) >= self.sequence_length:
            input_states = states[-self.sequence_length:]
        else:
            # Pad with zeros if not enough history
            padding = np.zeros((self.sequence_length - len(states), states.shape[1]))
            input_states = np.vstack([padding, states])
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_states).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            action_logits = outputs['action_logits']
            state_value = outputs['state_value']
            
            # Get probabilities
            action_probs = torch.softmax(action_logits, dim=1)
            predicted_action = torch.argmax(action_logits, dim=1).item()
            
            # Convert action to human-readable format
            action_names = ['Hold', 'Buy', 'Sell']
            
            result = {
                'predicted_action': action_names[predicted_action],
                'action_probabilities': {
                    'Hold': action_probs[0][0].item(),
                    'Buy': action_probs[0][1].item(),
                    'Sell': action_probs[0][2].item()
                },
                'estimated_value': state_value.item(),
                'confidence': torch.max(action_probs).item()
            }
            
            return result
    
    def analyze_decision_patterns(self, activities: List[Dict]) -> Dict:
        """Analyze decision patterns in the data"""
        df = self.state_encoder.extract_market_features(activities)
        
        analysis = {
            'total_activities': len(activities),
            'action_distribution': {},
            'price_action_correlation': {},
            'time_patterns': {},
            'market_patterns': {},
            'profitability_analysis': {}
        }
        
        # Action distribution
        actions = [a.get('side', 'unknown') for a in activities]
        action_counts = pd.Series(actions).value_counts()
        analysis['action_distribution'] = action_counts.to_dict()
        
        # Price-action correlation
        buy_prices = [float(a.get('price', 0)) for a in activities if a.get('side') == 'BUY']
        sell_prices = [float(a.get('price', 0)) for a in activities if a.get('side') == 'SELL']
        
        if buy_prices and sell_prices:
            analysis['price_action_correlation'] = {
                'avg_buy_price': np.mean(buy_prices),
                'avg_sell_price': np.mean(sell_prices),
                'buy_price_std': np.std(buy_prices),
                'sell_price_std': np.std(sell_prices)
            }
        
        # Time patterns
        hours = [pd.to_datetime(a.get('timestamp', '')).hour for a in activities if a.get('timestamp')]
        if hours:
            hour_counts = pd.Series(hours).value_counts().sort_index()
            analysis['time_patterns'] = {
                'most_active_hour': hour_counts.idxmax(),
                'hourly_distribution': hour_counts.to_dict()
            }
        
        # Market patterns
        markets = [a.get('title', '') for a in activities]
        market_counts = pd.Series(markets).value_counts()
        analysis['market_patterns'] = {
            'most_traded_market': market_counts.index[0] if not market_counts.empty else None,
            'market_distribution': market_counts.head(10).to_dict()
        }
        
        return analysis
    
    def visualize_training_history(self):
        """Visualize training metrics"""
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.training_history['loss'], label='Training Loss')
        ax1.plot(self.training_history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.training_history['accuracy'], label='Training Accuracy')
        ax2.plot(self.training_history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained model and encoders"""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'sequence_length': self.sequence_length,
            'state_encoder': self.state_encoder,
            'training_history': self.training_history
        }
        
        torch.save(save_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and encoders"""
        save_data = torch.load(filepath, map_location=self.device)
        
        self.model_type = save_data['model_type']
        self.sequence_length = save_data['sequence_length']
        self.state_encoder = save_data['state_encoder']
        self.training_history = save_data['training_history']
        
        # Recreate model architecture
        input_dim = len(self.state_encoder.feature_columns)
        self.model = self.create_model(input_dim)
        self.model.load_state_dict(save_data['model_state_dict'])
        
        print(f"Model loaded from {filepath}")


def main():
    """Example usage of the Markov Decision Reverse Engineering system"""
    
    # This would typically be called with real activity data from polymarket_monitor.py
    print("Markov Decision Process Reverse Engineering System")
    print("=" * 50)
    
    # Example: Load activities from a JSON file or API
    # activities = load_activities_from_file('activities.json')
    
    # For demonstration, create some synthetic data
    activities = create_synthetic_activities(1000)
    
    # Initialize the reverse engineering system
    mdp_system = MarkovDecisionReverseEngineer(model_type='lstm', sequence_length=10)
    
    # Train the model
    mdp_system.train(activities, epochs=50, batch_size=16)
    
    # Analyze decision patterns
    patterns = mdp_system.analyze_decision_patterns(activities)
    print("\nDecision Pattern Analysis:")
    for key, value in patterns.items():
        print(f"{key}: {value}")
    
    # Make predictions on recent activities
    recent_activities = activities[-20:]  # Last 20 activities
    prediction = mdp_system.predict_next_action(recent_activities)
    
    print(f"\nNext Action Prediction:")
    print(f"Predicted Action: {prediction['predicted_action']}")
    print(f"Confidence: {prediction['confidence']:.3f}")
    print(f"Action Probabilities: {prediction['action_probabilities']}")
    
    # Visualize training
    mdp_system.visualize_training_history()
    
    # Save the model
    mdp_system.save_model('markov_decision_model.pth')
    

def create_synthetic_activities(n: int) -> List[Dict]:
    """Create synthetic activity data for testing"""
    activities = []
    
    base_time = datetime.now()
    outcomes = ['up', 'down']
    sides = ['BUY', 'SELL']
    markets = ['Market A', 'Market B', 'Market C']
    
    for i in range(n):
        activity = {
            'timestamp': (base_time - timedelta(minutes=i)).isoformat(),
            'type': 'TRADE',
            'side': np.random.choice(sides),
            'title': np.random.choice(markets),
            'outcome': np.random.choice(outcomes),
            'size': np.random.uniform(100, 1000),
            'usdcSize': np.random.uniform(50, 500),
            'price': np.random.uniform(0.3, 0.7),
            'transactionHash': f'0x{i:064x}'
        }
        activities.append(activity)
    
    return activities


if __name__ == "__main__":
    main()
