
# Quantum Logistics Simulator

このプロジェクトは、**量子アニーリング（QUBO）**を用いて、複数の配送エージェント（トラック）の最適ルートをリアルタイムに計算・シミュレーションするデモアプリケーションです。

## 主な機能
- **Two-Phase QUBO VRP Solver**: 「誰がどの荷物を担当するか」と「どの順序で回るか」を2段階のQUBOで最適化します。
- **Real-time Multiplayer**: WebSocketを使用し、ブラウザ上で複数のクライアント（プレイヤー）とCPUが同じマップで動作します。
- **Logistics Dashboard UI**: 配送センターの管制モニターを模したUIで、配送進捗やフリートの状態を可視化します。
- **Guaranteed Delivery & Return**: 全ての配送先を確実に訪問し、完了後は全員がデポへ帰還します。

## 環境構築 (Setup)

Python 3.8以上が必要です。

### 1. ディレクトリへの移動
```bash
cd delivery_demo
```

### 2. 仮想環境の作成（推奨）
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# .\venv\Scripts\activate  # Windows
```

### 3. 依存ライブラリのインストール
```bash
pip install -r requirements.txt
```

## 起動方法 (Run)

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

ブラウザで `http://localhost:8000` にアクセスしてください。

## 遊び方 (How to Play)

1. **Lobby (待機画面)**:
   - **Map Scale**: ノード数を調整します。
   - **Delivery Targets**: 配送先の数を調整します（最大20）。
   - **Opponent AI Fleet**: CPUエージェント（自動運転）の数を調整します。
   - **Deploy Fleet**: ゲームを開始します。

2. **Simulation (シミュレーション)**:
   - 自分（You）はクリックした場所に移動できます。
   - CPUエージェントは自動で最適ルートを辿ります。
   - 配送完了後、全員がデポへ帰還します。

## 使用技術スタック (Tech Stack)

| カテゴリ | 技術 |
|---------|------|
| **バックエンド** | FastAPI (Python) |
| **フロントエンド** | Vanilla JavaScript + HTML5 Canvas |
| **量子アニーリング** | OpenJij (SA Sampler) + Dimod (BQM) |
| **通信** | WebSocket (リアルタイム双方向) |
| **主要ライブラリ** | NumPy, Uvicorn |

## アルゴリズム詳細: 二段階QUBO

### Phase 1: Assignment QUBO (割り当て)
- **変数**: $y_{k,i}$ (ドライバー $k$ が目的地 $i$ を担当するか)
- **制約**: 各目的地は必ず1人に割り当て (One-Hot)
- **目的**: 距離最小化 + 負荷分散

### Phase 2: TSP QUBO (巡回)
- **変数**: $z_{i,p}$ (目的地 $i$ を順序 $p$ で訪問するか)
- **制約**: Row/Column One-Hot (全目的地を1回ずつ訪問)
- **目的**: 総移動距離最小化（帰還コスト込み）

この二段階アプローチにより、全配送先への確実な訪問とデポへの帰還が数学的に保証されます。
