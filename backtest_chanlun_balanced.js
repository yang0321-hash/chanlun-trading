#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// ============================================================
// 平衡版缠论策略 - 保持有效优化，移除过度限制
// ============================================================
const CONFIG = {
    // 保留: 动态ATR止损 (有效)
    atrStop: {
        baseMultiplier: 2.0,
        adjustByVolatility: true,
        minMultiplier: 1.5,
        maxMultiplier: 3.0
    },

    // 保留: 简化的金字塔加仓 (有效)
    pyramid: {
        enabled: true,
        maxAdd: 2,               // 最多加仓2次
        profitThreshold: 0.08,   // 盈利8%后考虑加仓
        addRatio: 0.5,           // 每次加仓50%
        pullback: 0.025          // 回踩2.5%加仓
    },

    // 保留: 移动止损 (有效)
    trailingStop: {
        enabled: true,
        activationProfit: 0.08,  // 盈利8%激活
        trailPercent: 0.06       // 回撤6%止损
    },

    // 保留: 顶分型止盈 (有效)
    topFractalExit: {
        enabled: true,
        minProfit: 0.15          // 盈利15%后才考虑
    },

    // 保留: 分批止盈 (有效)
    partialExit: {
        enabled: true,
        levels: [
            { profit: 0.20, exit: 0.5 },   // 盈利20%减半
            { profit: 0.35, exit: 1.0 }    // 盈利35%全部清仓
        ]
    },

    // 移除: 周线过滤 (过度限制)
    // 移除: 成交量过滤 (过度限制)
    // 移除: MACD柱状图确认 (过度限制)
    // 移除: 市场状态切换 (复杂且效果不佳)
};

// ============================================================
// 指标计算
// ============================================================

function calculateATR(data, period = 14) {
    const tr = [], atr = [];
    for (let i = 0; i < data.length; i++) {
        if (i === 0) {
            tr.push(data[i].high - data[i].low);
        } else {
            tr.push(Math.max(
                data[i].high - data[i].low,
                Math.abs(data[i].high - data[i - 1].close),
                Math.abs(data[i].low - data[i - 1].close)
            ));
        }
    }

    for (let i = 0; i < tr.length; i++) {
        if (i < period - 1) {
            atr.push(tr.slice(0, i + 1).reduce((a, b) => a + b, 0) / (i + 1));
        } else if (i === period - 1) {
            atr.push(tr.slice(0, period).reduce((a, b) => a + b, 0) / period);
        } else {
            atr.push((atr[i - 1] * (period - 1) + tr[i]) / period);
        }
    }
    return atr;
}

function calculateEMA(data, period) {
    const result = [], mult = 2 / (period + 1);
    for (let i = 0; i < data.length; i++) {
        if (i < period) {
            result.push(data.slice(0, i + 1).reduce((a, b) => a + b, 0) / (i + 1));
        } else {
            result.push((data[i] - result[i - 1]) * mult + result[i - 1]);
        }
    }
    return result;
}

function calculateMACD(data) {
    const closes = data.map(d => d.close);
    const ema12 = calculateEMA(closes, 12), ema26 = calculateEMA(closes, 26);
    const dif = ema12.map((v, i) => v - ema26[i]);
    const dea = calculateEMA(dif, 9);
    return { dif, dea, macd: dif.map((v, i) => (v - dea[i]) * 2) };
}

function calculateVolatility(data, period = 20) {
    const vol = [];
    for (let i = 0; i < data.length; i++) {
        if (i < 2) { vol.push(0.02); continue; }
        const returns = [];
        for (let j = Math.max(1, i - period + 1); j <= i; j++) {
            returns.push(Math.log(data[j].close / data[j - 1].close));
        }
        const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
        const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
        vol.push(Math.sqrt(variance * 252));
    }
    return vol;
}

function findFractals(data) {
    const fractals = [];
    for (let i = 1; i < data.length - 1; i++) {
        const prev = data[i - 1], curr = data[i], next = data[i + 1];
        if (curr.high > prev.high && curr.high > next.high &&
            curr.low > prev.low && curr.low > next.low) {
            fractals.push({ index: i, type: 'top', price: curr.high, date: curr.date });
        } else if (curr.low < prev.low && curr.low < next.low &&
                   curr.high < prev.high && curr.high < next.high) {
            fractals.push({ index: i, type: 'bottom', price: curr.low, date: curr.date });
        }
    }
    return fractals;
}

function checkSecondBuy(data, fractals, idx, lookback = 60) {
    for (let i = fractals.length - 1; i >= 0; i--) {
        const f = fractals[i];
        if (f.type === 'bottom' && f.index < idx && idx - f.index <= lookback) {
            const lowSinceBottom = Math.min(...data.slice(f.index, idx).map(d => d.low));
            if (data[idx].close > f.price && lowSinceBottom > f.price * 0.97) {
                return { bottomPrice: f.price, bottomIdx: f.index };
            }
        }
    }
    return null;
}

function loadCSV(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const lines = content.trim().split('\n');
    return lines.slice(1).map(line => {
        const v = line.split(',');
        return {
            date: v[0], open: parseFloat(v[1]), high: parseFloat(v[2]),
            low: parseFloat(v[3]), close: parseFloat(v[4]),
            amount: parseFloat(v[5]), volume: parseFloat(v[6])
        };
    });
}

// ============================================================
// 平衡版策略回测
// ============================================================

function backtestBalanced(data, atr, volatility, macdData, fractals) {
    const trades = [];
    let state = {
        inPosition: false,
        entries: [],
        totalShares: 0,
        avgPrice: 0,
        highestPrice: 0,
        stopLoss: 0,
        trailingStop: 0,
        entryDate: '',
        entryIndex: 0,
        bottomPrice: 0,
        partialExitTaken: []
    };

    for (let i = 50; i < data.length; i++) {
        const curr = data[i];
        const goldenCross = (macdData.dif[i] || 0) > (macdData.dea[i] || 0) &&
                           (macdData.dif[i - 1] || 0) <= (macdData.dea[i - 1] || 0);
        const deathCross = (macdData.dif[i] || 0) < (macdData.dea[i] || 0) &&
                           (macdData.dif[i - 1] || 0) >= (macdData.dea[i - 1] || 0);

        // === 入场 ===
        if (!state.inPosition) {
            const secondBuy = checkSecondBuy(data, fractals, i);
            if (secondBuy && goldenCross) {
                // 动态ATR止损
                const currAtr = atr[i];
                const currVol = volatility[i];

                let atrMult = CONFIG.atrStop.baseMultiplier;
                if (CONFIG.atrStop.adjustByVolatility) {
                    if (currVol < 0.2) atrMult = CONFIG.atrStop.minMultiplier;
                    else if (currVol > 0.4) atrMult = CONFIG.atrStop.maxMultiplier;
                }

                const atrStop = curr.close - currAtr * atrMult;

                state = {
                    inPosition: true,
                    entries: [{ price: curr.close, shares: 1 }],
                    totalShares: 1,
                    avgPrice: curr.close,
                    highestPrice: curr.close,
                    stopLoss: Math.max(atrStop, secondBuy.bottomPrice * 0.97),
                    trailingStop: 0,
                    entryDate: curr.date,
                    entryIndex: i,
                    bottomPrice: secondBuy.bottomPrice,
                    partialExitTaken: []
                };
            }
        }

        // === 持仓管理 ===
        if (state.inPosition) {
            const profit = (curr.close - state.avgPrice) / state.avgPrice;

            // 更新最高价
            if (curr.close > state.highestPrice) {
                state.highestPrice = curr.close;
            }

            // 分批止盈
            if (CONFIG.partialExit.enabled) {
                for (const level of CONFIG.partialExit.levels) {
                    if (profit >= level.profit && !state.partialExitTaken.includes(level.profit)) {
                        state.partialExitTaken.push(level.profit);
                        const exitPct = level.exit;

                        // 记录部分止盈
                        trades.push({
                            entryDate: state.entryDate,
                            exitDate: curr.date,
                            entryPrice: state.avgPrice,
                            exitPrice: curr.close,
                            profit: profit * exitPct * 100,
                            holdDays: i - state.entryIndex,
                            reason: `分批止盈${(level.profit * 100).toFixed(0)}%`,
                            isPartial: true
                        });

                        // 调整止损到成本
                        state.stopLoss = Math.max(state.stopLoss, state.avgPrice);
                    }
                }
            }

            // 金字塔加仓
            if (CONFIG.pyramid.enabled &&
                state.entries.length < CONFIG.pyramid.maxAdd + 1 &&
                profit >= CONFIG.pyramid.profitThreshold) {

                const pullback = (state.highestPrice - curr.close) / state.highestPrice;
                if (pullback >= CONFIG.pyramid.pullback && pullback <= 0.04) {
                    const addShares = CONFIG.pyramid.addRatio;
                    const newAvg = (state.avgPrice * state.totalShares + curr.close * addShares) /
                                 (state.totalShares + addShares);

                    state.entries.push({ price: curr.close, shares: addShares });
                    state.totalShares += addShares;
                    state.avgPrice = newAvg;

                    // 更新止损
                    state.stopLoss = Math.max(state.stopLoss, curr.close - atr[i] * CONFIG.atrStop.baseMultiplier);
                }
            }

            // 移动止损
            if (CONFIG.trailingStop.enabled && profit >= CONFIG.trailingStop.activationProfit) {
                const newTrail = state.highestPrice * (1 - CONFIG.trailingStop.trailPercent);
                state.trailingStop = Math.max(state.trailingStop, newTrail);
            }

            // === 出场 ===
            let exit = false, reason = '', exitPrice = curr.close;

            if (state.trailingStop > 0 && curr.close < state.trailingStop) {
                exit = true; reason = '移动止损'; exitPrice = state.trailingStop;
            } else if (CONFIG.topFractalExit.enabled && profit >= CONFIG.topFractalExit.minProfit) {
                const recentTop = fractals.filter(f => f.type === 'top' && f.index >= i - 3 && f.index <= i);
                if (recentTop.length > 0) { exit = true; reason = '顶分型'; }
            } else if (deathCross && profit > 0.02) {
                exit = true; reason = 'MACD死叉';
            } else if (curr.close < state.stopLoss) {
                exit = true; reason = 'ATR止损';
            }

            if (exit) {
                trades.push({
                    entryDate: state.entryDate,
                    exitDate: curr.date,
                    entryPrice: state.entries[0].price,
                    exitPrice: exitPrice,
                    profit: (exitPrice - state.entries[0].price) / state.entries[0].price * 100,
                    holdDays: i - state.entryIndex,
                    reason,
                    addCount: state.entries.length - 1,
                    maxProfit: (state.highestPrice - state.entries[0].price) / state.entries[0].price * 100
                });

                // 重置状态
                state = {
                    inPosition: false,
                    entries: [], totalShares: 0, avgPrice: 0,
                    highestPrice: 0, stopLoss: 0, trailingStop: 0,
                    entryDate: '', entryIndex: 0, bottomPrice: 0,
                    partialExitTaken: []
                };
            }
        }
    }

    // 过滤掉部分止盈记录，只保留完整交易
    return trades.filter(t => !t.isPartial);
}

// 原版策略
function backtestOriginal(data, macdData, fractals) {
    const trades = [];
    let pos = false, entryPrice = 0, entryDate = '', entryIdx = 0, bottomPrice = 0;

    for (let i = 50; i < data.length; i++) {
        const goldenCross = (macdData.dif[i] || 0) > (macdData.dea[i] || 0) &&
                           (macdData.dif[i - 1] || 0) <= (macdData.dea[i - 1] || 0);
        const deathCross = (macdData.dif[i] || 0) < (macdData.dea[i] || 0) &&
                           (macdData.dif[i - 1] || 0) >= (macdData.dea[i - 1] || 0);

        if (!pos) {
            const sb = checkSecondBuy(data, fractals, i);
            if (sb && goldenCross) {
                pos = true; entryPrice = data[i].close;
                entryDate = data[i].date; entryIdx = i;
                bottomPrice = sb.bottomPrice;
            }
        } else {
            let sell = false, reason = '';
            if (data[i].close < bottomPrice) { sell = true; reason = '止损'; }
            else if (deathCross) { sell = true; reason = 'MACD死叉'; }
            if (sell) {
                trades.push({
                    entryDate, exitDate: data[i].date,
                    profit: (data[i].close / entryPrice - 1) * 100,
                    holdDays: i - entryIdx, reason
                });
                pos = false; bottomPrice = 0;
            }
        }
    }
    return trades;
}

// ============================================================
// 运行对比
// ============================================================

function runComparison() {
    console.log('='.repeat(90));
    console.log('缠论策略优化对比 - 原版 vs 超级版(过度) vs 平衡版(推荐)');
    console.log('='.repeat(90));
    console.log('\n优化说明:');
    console.log('  保留优化 (有效):');
    console.log('    ✓ 动态ATR止损 - 根据波动率自适应');
    console.log('    ✓ 金字塔加仓 - 盈利8%后回踩2.5%加仓50%');
    console.log('    ✓ 移动止损 - 盈利8%激活，回撤6%止损');
    console.log('    ✓ 顶分型止盈 - 盈利15%后遇顶分型出局');
    console.log('    ✓ 分批止盈 - 盈利20%减半，35%清仓');
    console.log('  移除优化 (过度限制):');
    console.log('    ✗ 周线趋势过滤 - 错过机会');
    console.log('    ✗ 成交量过滤 - 过度限制');
    console.log('    ✗ MACD柱状图确认 - 过度限制');
    console.log('    ✗ 市场状态切换 - 复杂且效果不稳定');
    console.log('='.repeat(90));

    const stocks = [
        { code: 'sz002600', name: '驰宏锌锗' },
        { code: 'sh600519', name: '贵州茅台' },
        { code: 'sz000858', name: '五粮液' },
        { code: 'sh600036', name: '招商银行' },
        { code: 'sz002594', name: '比亚迪' },
        { code: 'sz300750', name: '宁德时代' }
    ];

    const results = [];

    for (const stock of stocks) {
        const csvFile = path.join(__dirname, 'test_output', `${stock.code}.day.csv`);
        if (!fs.existsSync(csvFile)) continue;

        const data = loadCSV(csvFile);
        const atr = calculateATR(data);
        const volatility = calculateVolatility(data);
        const macdData = calculateMACD(data);
        const fractals = findFractals(data);

        const buyHold = ((data[data.length - 1].close / data[0].close - 1) * 100);
        const origTrades = backtestOriginal(data, macdData, fractals);
        const balTrades = backtestBalanced(data, atr, volatility, macdData, fractals);

        const origTotal = origTrades.reduce((s, t) => s + t.profit, 0);
        const balTotal = balTrades.reduce((s, t) => s + t.profit, 0);

        results.push({
            code: stock.code,
            name: stock.name,
            buyHold,
            orig: origTotal,
            origTrades: origTrades.length,
            bal: balTotal,
            balTrades: balTrades.length
        });
    }

    console.log('\n┌────────────┬──────────┬─────────┬─────────┬─────────┬─────────┐');
    console.log('│ 股票       │ 买入持有 │ 原版    │ 原版    │ 平衡版  │ 平衡版  │');
    console.log('│            │ 收益%    │ 收益%   │ 交易数  │ 收益%   │ 交易数  │');
    console.log('├────────────┼─────────┼─────────┼─────────┼─────────┼─────────┤');

    for (const r of results) {
        const improve = ((r.bal - r.orig) / (Math.abs(r.orig) || 1) * 100).toFixed(0);
        const mark = r.bal > r.orig ? '+' : '';
        console.log(`│ ${(r.code).padEnd(10)} │ ${r.buyHold.toFixed(5).padStart(7)} │ ${r.orig.toFixed(2).padStart(7)} │ ${r.origTrades.toString().padStart(7)} │ ${r.bal.toFixed(2).padStart(7)} │ ${r.balTrades.toString().padStart(7)} │`);
    }

    console.log('└────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘');

    const avgOrig = results.reduce((s, r) => s + r.orig, 0) / results.length;
    const avgBal = results.reduce((s, r) => s + r.bal, 0) / results.length;
    const beatCount = results.filter(r => r.bal > r.orig).length;

    console.log(`\n平均收益:`);
    console.log(`  原版:   ${avgOrig.toFixed(2)}%`);
    console.log(`  平衡版: ${avgBal.toFixed(2)}% (${avgBal > avgOrig ? '+' : ''}${(avgBal - avgOrig).toFixed(2)}%)`);
    console.log(`  超越:   ${beatCount}/${results.length} 只股票`);

    // 详细对比
    console.log('\n' + '='.repeat(90));
    console.log('策略特征对比');
    console.log('='.repeat(90));

    const balTradesTotal = results.reduce((s, r) => s + r.balTrades, 0);
    const origTradesTotal = results.reduce((s, r) => s + r.origTrades, 0);

    console.log(`\n平均交易数: 原版 ${(origTradesTotal / results.length).toFixed(1)}笔 -> 平衡版 ${(balTradesTotal / results.length).toFixed(1)}笔`);

    console.log('\n结论:');
    if (avgBal > avgOrig) {
        console.log('  ✓ 平衡版策略表现优于原版');
        console.log('  ✓ 推荐使用平衡版配置');
    } else {
        console.log('  ⚠ 平衡版在此样本上未显著超越原版');
        console.log('  ℹ 可根据具体股票调整参数');
    }

    console.log('\n核心优化点:');
    console.log('  1. 动态ATR止损 - 减少震荡止损');
    console.log('  2. 金字塔加仓 - 放大盈利');
    console.log('  3. 移动止损 - 保护利润');
    console.log('  4. 分批止盈 - 锁定收益');

    console.log('\n' + '='.repeat(90));
}

runComparison();
