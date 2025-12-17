
# Quantum Logistics Simulator

このプロジェクトは、**量子アニーリング（QUBO）**を用いて、複数の配送エージェント（トラック）の最適ルートをリアルタイムに計算・シミュレーションするデモアプリケーションです。

古典的な「エリア分け（クラスタリング）」と「巡回セールスマン問題（TSP）」を分離せず、**単一の巨大なQUBOモデル**として定式化し、全体最適解（Vehicle Routing Problem: VRP）を導き出します。

## 主な機能
- **Full QUBO VRP Solver**: エージェントの「担当エリア決定」と「訪問順序」を同時に最適化します。
- **Real-time Multiplayer**: WebSocketを使用し、ブラウザ上で複数のクライアント（プレイヤー）とCPUが同じマップで動作します。
- **Logistics Dashboard UI**: 配送センターの管制モニターを模したUIで、配送進捗やフリートの状態を可視化します。

## 環境構築 (Setup)

Python 3.8以上が必要です。

### 1. ディレクトリへの移動
```bash
cd q_compe/delivery_demo
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
※ `openjij`, `dimod`, `fastapi`, `uvicorn`, `numpy` 等がインストールされます。

## 起動方法 (Run)

以下のコマンドでサーバーを起動します。

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

起動したら、ブラウザで以下にアクセスしてください。

- **URL**: `http://localhost:8000`
- （同じWi-Fi内のスマホ等からアクセスする場合: `http://<PCのIPアドレス>:8000`）

## 遊び方 (How to Play)

1. **Lobby (待機画面)**:
   - 接続するとロビー画面になります。
   - **"Redesign Network"**: マップ（配送先とデポ）をランダムに再生成します。
   - **"Opponent AI Fleet"**: CPUエージェント（自動運転）の数を調整します。
   - **"Deploy Fleet"**: ゲームを開始します。接続中の全プレイヤーが一斉にスタートします。

2. **Simulation (シミュレーション)**:
   - 自分（You）はクリックした場所に移動できます。
   - CPUエージェントは自動で最適ルートを辿ります。
   - 裏側では「誰かが動く」たびに量子アニーリング（OpenJij）が走り、全員の残り配送先に基づいて最適なルートを再計算して指示を出します。

3. **Status**:
   - 左側のサイドバーで配送進捗率（Delivery Progress）を確認できます。

## 使用技術スタック (Tech Stack)

*   **バックエンド**: FastAPI (Python)
    *   非同期処理による高速なWebサーバーとWebSocket通信の実装に使用。
*   **フロントエンド**: Vanilla JavaScript + HTML5 Canvas
    *   フレームワーク（React等）への依存を排除し、多数のエージェントをリアルタイムに描画するための軽量実装。
*   **量子アニーリング / 最適化**: OpenJij, dimod
    *   QUBOモデルの構築とシミュレーテッド・アニーリングによる解探索に使用。
*   **主要ライブラリ**: NumPy, Uvicorn, Websockets

### アルゴリズム詳細
- **Multi-Agent Vehicle Routing Problem (VRP) formulated as QUBO.**
  - 変数: $x_{k,t,i}$ (エージェント $k$ が ステップ $t$ で 目的地 $i$ を訪問するか)
  - 制約: 各目的地は必ず1回訪問される / 同時刻に1人のエージェントは1箇所しか訪問できない 等
