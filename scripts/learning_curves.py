"""Create a small, dependency-free HTML snapshot from existing training logs."""

# ruff: noqa: E501 -- Embedded standalone HTML/JS template.

import argparse
import datetime
import json
from pathlib import Path


def load_data(run):
    rows = []
    for line in (run / "metrics.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    train = []
    for i, row in enumerate(rows):
        episodes = row["episodes"]
        window = rows[max(0, i - 9) : i + 1]
        pooled = [e for r in window for e in r["episodes"]]
        count = max(len(pooled), 1)
        train.append(
            dict(
                x=row["update"],
                ticks=row["logic_ticks"],
                score=sum(e["score"] for e in episodes) / max(len(episodes), 1),
                smooth_score=sum(e["score"] for e in pooled) / count,
                lifetime=sum(e["ticks"] for e in pooled) / count,
                wins=100 * sum(e["won"] for e in pooled) / count,
                value_loss=row["value_loss"],
                kl=row["approx_kl"],
                sil_rows=row.get("sil_rows", 0),
                sil_valid=row["sil_valid"],
            )
        )
    evaluation = []
    for path in sorted(run.glob("validation-*.json")):
        try:
            result = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        episodes = result["episodes"]
        n = len(episodes)
        evaluation.append(
            dict(
                x=int(path.name.split("-")[1].split(".")[0]),
                mode="确定性" if result["deterministic_policy"] else "采样",
                n=n,
                score=sum(e["score"] for e in episodes) / n,
                lifetime=sum(e["ticks"] for e in episodes) / n,
                wins=100 * sum(e["won"] for e in episodes) / n,
                maximum=max(e["score"] for e in episodes),
            )
        )
    return dict(
        train=train,
        evaluation=evaluation,
        source=str(run),
        generated=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


HTML = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>31×20 贪吃蛇 · 学习曲线</title>
<style>
body{font:16px/1.7 system-ui,sans-serif;color:#253344;background:#f4f6f8;margin:0}
main{max-width:1160px;margin:auto;padding:30px}h1{font-size:27px;margin-bottom:4px}
h2{font-size:18px;margin:0 0 8px}p{margin:8px 0}.note{color:#64748b;font-size:14px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:22px 0}
section,.summary{background:white;border:1px solid #dce3ea;border-radius:10px;padding:20px}
svg{width:100%;height:auto}text{font:12px system-ui;fill:#64748b}.legend{font-size:13px}
.legend span{margin-right:16px}.dot{display:inline-block;width:10px;height:10px;margin-right:5px}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:right;padding:7px;border-bottom:1px solid #e5e9ed}
th:first-child,td:first-child{text-align:left}.source{overflow-wrap:anywhere}
@media(max-width:750px){.grid{grid-template-columns:1fr}main{padding:15px}}
</style><main><h1>31×20 贪吃蛇：训练是否在进步？</h1>
<p class="note">run04 · 完整棋盘、手柄动作、随机延迟与扰动 · 静态快照，不会自动刷新</p>
<div class="summary" id="summary"></div>
<p>主要看独立验证的取食数和通关率是否持续上升，存活时间作为辅助。只活得更久可能是在绕圈；loss 下降不等于学会玩。</p>
<div class="grid">
<section><h2>训练中每局吃到的食物</h2><div id="train"></div><p class="note">浅线：当轮均值；深线：最近10轮已结束游戏的合并均值。不同长度回合按局计，不把每轮均值再平均。</p></section>
<section><h2>独立验证：每局食物数</h2><div id="score"></div><p class="note">固定验证种子用于诊断，不是最终未见测试集。每点30局；低分的小幅波动不能证明提升。</p></section>
<section><h2>独立验证：存活游戏 tick</h2><div id="lifetime"></div><p class="note">游戏逻辑步，不是墙钟运行时间。必须结合取食曲线看。</p></section>
<section><h2>独立验证：通关率（%）</h2><div id="wins"></div><p class="note">目标是在31×20场地占满可用格；曲线为0表示本批未观察到通关，不表示真实成功概率严格为0。</p></section>
<section><h2>价值损失（诊断项）</h2><div id="loss"></div><p class="note">估值误差变小不等于策略变好；回报分布变化也会改变损失。</p></section>
<section><h2>自主经验回放利用量</h2><div id="sil"></div><p class="note">每轮回放决策数与正优势决策数，含重复抽样。它只说明数据是否参与优化，不是成功率。</p></section>
</div>
<section><h2>验证原始汇总</h2><table><thead><tr><th>续训轮次</th><th>策略</th><th>局数</th><th>平均食物</th><th>最高食物</th><th>平均存活tick</th><th>通关率</th></tr></thead><tbody id="rows"></tbody></table></section>
<p class="note">横轴均为本次续训轮次，从run02第60轮检查点继续；不是从零训练。图中保留零基线，没有截断纵轴来放大细小进步。</p>
<p class="note source" id="source"></p>
<script id="data" type="application/json">__DATA__</script>
<script>
const d=JSON.parse(document.getElementById('data').textContent);
const blue='#176caa', orange='#c77815', faint='#c6d7e6';
function chart(id,series,minimumMax=0){
 const all=series.flatMap(s=>s.points), W=510,H=250,L=55,R=15,T=18,B=35;
 if(!all.length){document.getElementById(id).textContent='尚无记录';return}
 const xmax=Math.max(1,...d.train.map(p=>p.x)), ymax=Math.max(minimumMax,1e-6,...all.map(p=>p.y))*1.08;
 const x=v=>L+v/xmax*(W-L-R), y=v=>H-B-v/ymax*(H-T-B);
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${id}学习曲线">`;
 for(let i=0;i<=4;i++){const v=ymax*i/4;svg+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="#e8edf1"/><text x="${L-7}" y="${y(v)+4}" text-anchor="end">${v<1?v.toFixed(2):v.toFixed(0)}</text>`;const t=Math.round(xmax*i/4);svg+=`<text x="${x(t)}" y="${H-12}" text-anchor="middle">${t}</text>`}
 for(const s of series){svg+=`<polyline fill="none" stroke="${s.color}" stroke-width="2" points="${s.points.map(p=>`${x(p.x)},${y(p.y)}`).join(' ')}"/>`;if(s.points.length<40)for(const p of s.points)svg+=`<circle cx="${x(p.x)}" cy="${y(p.y)}" r="3" fill="${s.color}"><title>轮次${p.x}: ${p.y.toFixed(4)}</title></circle>`}
 document.getElementById(id).innerHTML=svg+'</svg><div class="legend">'+series.map(s=>`<span><i class="dot" style="background:${s.color}"></i>${s.name}</span>`).join('')+'</div>';
}
const train=(key,name,color)=>({name,color,points:d.train.map(p=>({x:p.x,y:p[key]}))});
const ev=key=>['确定性','采样'].map((mode,i)=>({name:mode,color:[blue,orange][i],points:d.evaluation.filter(p=>p.mode===mode).map(p=>({x:p.x,y:p[key]}))}));
chart('train',[train('score','当轮',faint),train('smooth_score','最近10轮',blue)]);
chart('score',ev('score'));chart('lifetime',ev('lifetime'));chart('wins',ev('wins'),100);
chart('loss',[train('value_loss','value loss',blue)]);
chart('sil',[train('sil_rows','回放决策',blue),train('sil_valid','正优势决策',orange)]);
const last=d.train.at(-1),maxEval=Math.max(0,...d.evaluation.map(p=>p.x));
document.getElementById('summary').textContent=`训练到第 ${last?.x??0} 轮，累计 ${last?.ticks?.toLocaleString()??0} 游戏tick（含续训前）。最新验证：第 ${maxEval} 轮。${d.evaluation.some(p=>p.wins>0)?'出现通关候选，仍需审计和未见种子复测。':'已记录的验证中尚无通关，不能称为学会。'}`;
document.getElementById('rows').innerHTML=d.evaluation.map(p=>`<tr><td>${p.x}</td><td>${p.mode}</td><td>${p.n}</td><td>${p.score.toFixed(3)}</td><td>${p.maximum}</td><td>${p.lifetime.toFixed(1)}</td><td>${p.wins.toFixed(1)}%</td></tr>`).join('');
document.getElementById('source').textContent=`来源：${d.source}；快照生成时间（UTC）：${d.generated}`;
</script></main></html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load_data(args.run)
    args.output.write_text(
        HTML.replace(
            "__DATA__", json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
        )
    )
    print(
        json.dumps(
            {"updates": len(data["train"]), "validation": data["evaluation"]},
            ensure_ascii=False,
        )
    )
