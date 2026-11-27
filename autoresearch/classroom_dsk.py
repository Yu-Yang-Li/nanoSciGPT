"""Run the corrected ATLAS-A classroom AI Scientist with the DSK client from package 06."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesson", required=True, type=Path, help="解压后的 04 ATLAS-A 实操包")
    parser.add_argument("--client-dir", required=True, type=Path, help="解压后的 06 DSK 接口包")
    parser.add_argument("--output", required=True, type=Path, help="新的实验输出目录")
    parser.add_argument("--writeup-from", type=Path, help="从已锁定的实验记录重写报告和复评，沿用全部实验结果")
    args = parser.parse_args()
    sys.path[:0] = [str(args.lesson.resolve()), str(args.client_dir.resolve())]
    import gravity_lesson as g
    import model_client

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    calls_dir = out/"calls"
    calls_dir.mkdir()
    start = time.perf_counter()
    http_attempts = []
    # urllib emits this event for each real request, including transport retries. Never record headers.
    def audit(event, data):
        if event == "urllib.Request":
            http_attempts.append(time.perf_counter())
    sys.addaudithook(audit)
    journal = {"status":"running", "calls":[], "stages":[]}
    def update():
        journal["elapsed_seconds"] = round(time.perf_counter()-start, 3)
        journal["http_attempts"] = len(http_attempts)
        journal["model_responses"] = sum("usage" in c for c in journal["calls"])
        journal["tokens"] = sum(c.get("usage",{}).get("total_tokens",0) for c in journal["calls"])
        dump(out/"receipt.json", journal)

    def call(stage, prompt, check):
        journal["active_stage"] = stage
        for attempt in (1,2):
            tag = f"{len(journal['calls'])+1:02d}-{stage}"
            (calls_dir/(tag+".prompt.txt")).write_text(prompt, encoding="utf-8")
            entry = {"stage":stage,"attempt":attempt,"prompt_file":f"calls/{tag}.prompt.txt"}
            journal["calls"].append(entry)
            update()
            before, tick = len(http_attempts), time.perf_counter()
            try:
                answer, meta = model_client.complete(prompt,max_tokens=2600)
                entry.update(meta)
                entry["response_file"]=f"calls/{tag}.response.txt"
                (out/entry["response_file"]).write_text(answer,encoding="utf-8")
                if meta["configured_model"]!=meta["returned_model"]:
                    raise RuntimeError("返回型号与配置不一致")
                clean=answer.strip()
                if clean.startswith("```"):
                    clean=clean.split("\n",1)[1].rsplit("```",1)[0].strip()
                value=json.loads(clean)
                check(value)
                entry["accepted"]=True
                return value
            except (ValueError,TypeError,KeyError) as exc:
                entry["accepted"]=False
                entry["error"]=str(exc)
                if attempt==2:
                    raise
                prompt += "\n上次回答的格式校验未通过。请修正这一点并返回完整 JSON：" + str(exc)
            finally:
                entry["wall_seconds"]=round(time.perf_counter()-tick,3)
                entry["http_attempts"]=len(http_attempts)-before
                update()

    def fields(value, names):
        if not isinstance(value,dict) or any(not isinstance(value.get(n),str) or not value[n].strip() for n in names):
            raise ValueError("以下字段必须为非空文本：" + ", ".join(names))

    def stage(name,tick,files):
        journal["stages"].append({"name":name,"seconds":round(time.perf_counter()-tick,3),"files":files})
        update()

    def write_review(evidence):
        tick=time.perf_counter()
        names=["title","question","methods","results","limitations"]
        confirm=evidence["confirmation"]
        base_rows=confirm["baseline"]["mae_by_seed"]
        trial_rows=confirm["locked"]["mae_by_seed"]
        def check_text(value):
            fields(value,names)
            prose=" ".join(value[k] for k in names)
            if any(re.search(r"(?<![\d.])"+str(seed)+r"(?!\d)",prose) for seed in base_rows):
                raise ValueError("请省略确认种子编号，程序按真实误差追加完整逐种子表")
        written=call("write","你是实验报告作者。只根据下方实测记录写一份简短中文课堂研究报告。给出起点与锁定 α、开发和确认 MAE、五组配对胜场。区分开发与确认，论断回扣起始假设。精确到小数点后5位即可。搜索方法以实测过程为准：三轮均由 DSK 阅读当轮之前的开发结果后生成。若起始想法与执行有差别，说明实际执行。请区分最终锁定和每轮暂时保留。仅汇总配对胜场；省略确认种子编号，程序会从原始误差追加逐种子表。不得虚构显著性、文献或额外实验。只返回 JSON，字段为 title, question, methods, results, limitations，均为中文文本。\n"+json.dumps(evidence,ensure_ascii=False),check_text)
        report="# "+written["title"]+"\n\n"+"\n\n".join("## "+h+"\n\n"+written[n] for n,h in zip(names[1:],["研究问题","方法","结果","局限"]))+"\n"
        report+="\n## 确认集逐种子核对\n\n由 CPU 评分回执生成，单位为 dex。\n\n| 标签种子 | 起点 MAE | 锁定 MAE | 起点减锁定 |\n|---|---:|---:|---:|\n"
        for seed,baseline_mae in base_rows.items():
            chosen_mae=trial_rows[seed]
            report+=f"| {seed} | {baseline_mae:.6f} | {chosen_mae:.6f} | {baseline_mae-chosen_mae:+.6f} |\n"
        (out/"report.md").write_text(report,encoding="utf-8")
        stage("短报告",tick,["writing_evidence.json","report.md"])
        tick=time.perf_counter()
        dims=["可检验性","对照设计","数字可追溯性","结论克制程度"]
        def valid_review(v):
            fields(v,["verdict","main_limitation"])
            for d in dims:
                item=v["scores"][d]
                fields(item,["reason"])
                if type(item.get("score")) is not int or not 0<=item["score"]<=5: raise ValueError(d+" 需要0至5的整数分")
        review=call("review","你是独立课堂审稿人。用新的上下文读报告并逐项核对下面的真实记录。按可检验性、对照设计、数字可追溯性、结论克制程度四维各给0至5整数，理由要指出具体依据。务必核对三轮提议来自 DSK 的实际记录，预注册的是门槛与预算。五组评分共享确认星，本轮开发星与确认星互斥。查找数值或因果归因错误，成绩按证据质量与写作质量评，不按是否胜过基线。返回中文 JSON：scores 的键为上述四个中文维度，每个值含 score、reason；另含 verdict、main_limitation 两个文本字段。\n报告：\n"+report+"\n实测记录：\n"+json.dumps(evidence,ensure_ascii=False),valid_review)
        review["total"]=sum(review["scores"][d]["score"] for d in dims)
        review["max_score"]=20
        review["report_sha256"]=sha(out/"report.md")
        dump(out/"review.json",review)
        (out/"review.md").write_text("# 课堂评审 "+str(review["total"])+"/20\n\n"+"\n\n".join(f"**{d} {review['scores'][d]['score']}/5**　{review['scores'][d]['reason']}" for d in dims)+"\n\n"+review["verdict"]+"\n\n"+review["main_limitation"]+"\n",encoding="utf-8")
        stage("独立上下文评审",tick,["review.json","review.md"])

    dev_seeds=[241,257,271,283,307]
    confirm_seeds=[401,419,431,443,457]
    protocol={"task":"ATLAS-A 恒星光谱表面重力回归","labels":32,"dev_stars":878,"confirm_stars":913,
              "dev_seeds":dev_seeds,"confirm_seeds":confirm_seeds,"alpha_initial":1.0,
              "candidate_budget":3,"threshold_dex":0.001,"alpha_range":[0.0625,64.0],
              "pretrain_steps":480,"finetune_steps":80,"cpu_threads":2,"metric":"五组标签抽样等权平均 MAE，单位 dex，越低越好",
              "script_sha256":sha(Path(__file__)),
              "inputs_sha256":{n:sha(args.lesson/n) for n in ("gravity.npz","gravity_lesson.py","nanoSciGPT-b9c3486.zip")}}
    if not args.writeup_from:
        dump(out/"protocol.json",protocol)
    public="ATLAS-A 光谱预测恒星表面重力。Gaia 辅助预训练480步，微调80步，每次32个标签。只改变微调前回归头的岭正则 α。开发集878颗星，比较相同五组标签抽样的平均MAE（dex，越低越好）；至少降低0.001才替换起点。起点α=1，三个新提议，由 DSK 逐轮查看当前开发记录后选取，范围[0.0625,64]。"
    try:
        if args.writeup_from:
            prior=args.writeup_from.resolve()
            load=lambda n:json.loads((prior/n).read_text(encoding="utf-8"))
            lock, confirm = load("lock.json"), load("confirmation.json")
            if (lock["protocol_sha256"]!=sha(prior/"protocol.json") or
                    lock["idea_sha256"]!=sha(prior/"idea.json") or confirm["lock_sha256"]!=sha(prior/"lock.json")):
                raise ValueError("想法、协议、锁定记录或确认回执哈希不匹配")
            evidence = load("writing_evidence.json")
            if evidence["development"]!=lock or evidence["idea"]!=load("idea.json"):
                raise ValueError("写作证据与实验记录不一致")
            for key in ("baseline","locked"):
                item=confirm[key]
                if evidence["confirmation"][key] != {"alpha":item["alpha"],"mean_mae":item["mean"],
                        "mae_by_seed":{str(r["seed"]):r["mae"] for r in item["rows"]}}:
                    raise ValueError("写作证据与确认误差不一致")
            for key in ("stars","gain","wins","ties"):
                if confirm[key] != evidence["confirmation"][key]:
                    raise ValueError("确认汇总与写作证据不一致")
            evidence["selection_method"]="DSK 每轮阅读已完成的开发记录，独立给出下一个数值与理由。预先固定的是预算、门槛和范围。"
            dump(out/"writing_evidence.json",evidence)
            journal["parent_artifacts"]={n:sha(prior/n) for n in ("idea.json","protocol.json","lock.json","confirmation.json","report.md","review.json")}
            journal["script_sha256"]=sha(Path(__file__))
            write_review(evidence)
            journal["status"]="completed"
            journal["active_stage"]=None
            return
        tick=time.perf_counter()
        idea=call("idea",public+"你是课堂 AI Scientist。写一个可证伪的研究想法，说明什么结果会削弱它。没有实验分数。返回中文 JSON，仅含 question、hypothesis、falsifier 三个文本字段。",
                  lambda v:fields(v,["question","hypothesis","falsifier"]))
        dump(out/"idea.json",idea)
        stage("想法",tick,["idea.json","protocol.json"])
        tick=time.perf_counter()
        torch.set_num_threads(2)
        encoder=g.load_encoder()
        _,source,y,gaia,x,train,paired,eligible,dev,test=g.load_spectra()
        assert len(dev)==878 and len(test)==913 and not np.intersect1d(dev,test).size
        def prepare(seeds):
            jobs=[]
            for seed in seeds:
                torch.manual_seed(seed)
                model=g.GravityNet(encoder(16,25,hidden_dim=32))
                model=g.pretrain_gaia(model,x[paired],gaia[paired],seed,steps=480)
                ids=np.sort(np.random.default_rng(seed).permutation(eligible)[:32])
                assert np.isin(ids,train).all()
                jobs.append((seed,model,ids))
            return jobs
        def score(alpha,jobs,stars):
            rows=[]
            with patch.object(g,"Ridge",lambda **kw:Ridge(**{**kw,"alpha":alpha})):
                for seed,model,ids in jobs:
                    pred=g.fit_gravity(model,x[ids],y[ids],x[stars],seed,pretrained=True)
                    if not np.isfinite(pred).all(): raise ValueError("nonfinite predictions")
                    rows.append({"seed":seed,"label_ids":g.id_hash(source[ids]),"mae":float(np.abs(pred-y[stars]).mean())})
            return {"alpha":alpha,"mean":float(np.mean([r["mae"] for r in rows])),"rows":rows}
        def summary(item):
            return {"alpha":item["alpha"],"mean_mae":item["mean"],"mae_by_seed":{r["seed"]:r["mae"] for r in item["rows"]}}
        with threadpool_limits(limits=2,user_api="blas"):
            dev_jobs=prepare(dev_seeds)
            baseline=score(1.0,dev_jobs,dev)
            dump(out/"dev_baseline.json",baseline)
            incumbent=baseline
            seen=[1.0]
            history=[]
            for rnd in range(1,4):
                state={"idea":idea,"round":rnd,"incumbent_alpha":incumbent["alpha"],"baseline":summary(baseline),"previous_trials":history}
                def valid_proposal(v):
                    fields(v,["reason"])
                    a=v.get("alpha")
                    if type(a) not in (int,float) or not math.isfinite(a) or not 0.0625<=a<=64 or any(math.isclose(a,k,rel_tol=1e-8) for k in seen):
                        raise ValueError("alpha 应是范围内未试过的新数值")
                prop=call(f"proposal-{rnd}",public+"你是候选提议者。根据以下实际开发记录决定下一个 α；理由指出记录中的具体现象和本次提议想检验什么。你自行选数值，禁用已试过的α。输出中文 JSON，字段 alpha（数值）和 reason（文本）。\n"+json.dumps(state,ensure_ascii=False),valid_proposal)
                alpha=float(prop["alpha"])
                seen.append(alpha)
                candidate=score(alpha,dev_jobs,dev)
                gain=incumbent["mean"]-candidate["mean"]
                event={"round":rnd,"reason":prop["reason"],"compared_with":incumbent["alpha"],"result":summary(candidate),"gain":gain,"accepted":gain>=0.001}
                dump(out/f"round-{rnd}.json",{**event,"paired_rows":candidate["rows"]})
                history.append(event)
                if event["accepted"]: incumbent=candidate
            lock={"protocol_sha256":sha(out/"protocol.json"),"idea_sha256":sha(out/"idea.json"),"dev_ids":g.id_hash(source[dev]),
                  "incumbent_alpha":incumbent["alpha"],"baseline":summary(baseline),"history":history}
            dump(out/"lock.json",lock)
            stage("开发实验与锁定",tick,["dev_baseline.json","round-1.json","round-2.json","round-3.json","lock.json"])
            tick=time.perf_counter()
            locked=json.loads((out/"lock.json").read_text(encoding="utf-8"))["incumbent_alpha"]
            fresh=prepare(confirm_seeds)
            control=score(1.0,fresh,test)
            chosen=score(locked,fresh,test) if locked!=1 else control
            delta=[a["mae"]-b["mae"] for a,b in zip(control["rows"],chosen["rows"])]
            assert [r["label_ids"] for r in control["rows"]]==[r["label_ids"] for r in chosen["rows"]]
            confirm={"lock_sha256":sha(out/"lock.json"),"confirm_ids":g.id_hash(source[test]),"stars":len(test),
                     "baseline":control,"locked":chosen,"gain":float(np.mean(delta)),"wins":sum(d>0 for d in delta),"ties":sum(d==0 for d in delta)}
            dump(out/"confirmation.json",confirm)
            stage("锁定后确认",tick,["confirmation.json"])
        evidence={"protocol":public,"idea":idea,"development":lock,"confirmation":{"baseline":summary(control),"locked":summary(chosen),
                  "stars":len(test),"gain":confirm["gain"],"wins":confirm["wins"],"ties":confirm["ties"]},
                  "boundary":"确认星与本轮开发星互斥，五次评分共享同一批确认星。该课程历史上曾使用这批星。评审为同一DSK型号的新上下文。"}
        dump(out/"writing_evidence.json",evidence)
        write_review(evidence)
        journal["status"]="completed"
        journal["active_stage"]=None
    except Exception as exc:
        journal["status"]="failed"
        journal["error_type"]=type(exc).__name__
        raise
    finally:
        update()


if __name__=="__main__":
    main()
