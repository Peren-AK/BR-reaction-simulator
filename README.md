# BR-reaction-simulator(回分反応器シミュレーション)

## はじめに
- 本リポジトリは秋山隼輝が独自開発した3DCGソフトBlender用のアドオンに関するものです。
- bpyと呼ばれるBlender Python BPIを使用しています。
- ご利用いただくことでのトラブル等は一切責任を負いかねます。

## コンセプト
- 私が大学で専攻している反応工学では反応速度式を頻繁に取り扱うのですが、式変形によってある一つのタイミングでの各数値のみを追うことが多く、その途中過程に目を向けることはほとんどありませんでした。これでは実際の現場に即した理解ができないと感じ、実際に反応器内でどのような変化が起こっているかを可視化することで私を含めた学習者の理解をより深めることができるのではないかと考え作成しました。
- 現時点ではBR(回分反応器)のみでのシミュレーションを実装しております。
- 初めて自力で完成させた制作物であり、未熟な部分が多いかもしれませんが温かい目でご覧いただければ幸いです。

## アドオン概要
- アドオンとしてBlenderに導入すると、サイドバーに「Reaction Sim」というシミュレーション設定をするタブが追加されます。
- 該当タブを操作することで、反応定数・初期濃度・各係数・次数を変更できます。
- 各粒子のマテリアルも「Material Menu」から簡単に変更できます。
- シミュレーションを開始すると、それまでのオブジェクト等は全てリセットされるので、繰り返しシミュレーションを行うことができます。

## デモ動画

https://github.com/user-attachments/assets/2e84d482-8810-4b4f-a937-52ca13802fd4

高解像度の動画はこちらからご覧いただけます。<br>
https://youtu.be/q2EHXChnyhk?si=dQ-ZVX1u0UC1Hn1h

## 環境
- 開発言語: Python 3.9.6
- アプリケーション: Blender 4.1.1
- 検証済みOS: macOS Sequoia 15.3.1

## 利用方法
- Blender上の編集タブ→プリファレンス→アドオン→インストールから「BR_simulator.py」ファイルを選択し、インストールした後チェックを入れてください。
- サイドバーに「Reaction Sim」のタブが追加されているので、そちらから反応定数・初期濃度等必要事項を各自選択し、「Run Simulator」ボタンを押すとシミュレーションが生成されます。
- 「Material Menu」をクリックするとダイアログが表示されるので、そこから各粒子の色を変更することができます。
- 「Brownian Motion」についてはチェックが入っていると、粒子がブラウン運動を再現した運動を行うようになり、チェックを外すと、座標は動かなくなります。
- 生成されてからは、タイムライン上からの操作、もしくはスペースキーによってアニメーションを動かすことができます。

## 制約
- Aが限定反応成分(反応終了時に濃度が0)となるように初期濃度・係数を定めないとエラーが発生します。

## 今後の実装予定の機能
- PFR(管型反応器)におけるシミュレーションの実装
- 循環流れによる各粒子の移動を描写

## おわりに
- 二ヶ月ほど前に始めたpython3学習のアウトプットとして、リポジトリ公開させていただきました。
- ご覧頂きありがとうございました！
