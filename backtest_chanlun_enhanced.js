#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// 优化参数配置
const CONFIG = {
    // ATR止损倍数
    atrStopMultiplier: 2.0,

    // 金字塔加仓配置
    pyramid: {
        enabled: true,
        maxPositions: 3,           // 最多3次加仓
        addThreshold: 0.05,        // 盈利5%后考虑加仓
        addRatio: 0.5,             // 每次加仓50%原始仓位
        pullbackRatio: 0.02        // 回踩2%时加仓
    },

    // 移动止损
    trailingStop: {
        enabled: true,
        activationProfit: 0.10,    // 盈利10%后激活
        trailPercent: 0.05         // 回撤5%止损
    },

    // 风险管理
    risk: {
        maxLossPerTrade: 0.05,     // 单笔最大亏损5%
        maxDrawdown: 0.20,         // 最大回撤20%停止交易
        positionSize: 1.0          // 初始仓位
    }
};

// 计算ATR (Average True Range) - 动态止损核心
function calculateATR(data, period = 14) {
    const tr = [];
    for (let i = 0; i < data.length; i++) {
        if (i === 0) {
            tr.push(data[i].high - data[i].low);
        } else {
            const hl = data[i].high - data[i].low;
            const hc = Math.abs(data[i].high - data[i - 1].close);
            const lc = Math.abs(data[i].low - data[i - 1].close);
            tr.push(Math.max(hl, hc, lc));
        }
    }

    const atr = [];
    for (let i = 0; i < tr.length; i++) {
        if (i < period - 1) {
            atr.push(tr[i]);
        } else if (i === period - 1) {
            const sum = tr.slice(0, period).reduce((a, b) => a + b, 0);
            atr.push(sum / period);
        } else {
            atr.push((atr[i - 1] * (period - 1) + tr[i]) / period);
        }
    }

    return atr;
}

// 计算MACD
function calculateMACD(data) {
    const closes = data.map(d => d.close);
    const ema12 = calculateEMA(closes, 12);
    const ema26 = calculateEMA(closes, 26);
    const dif = ema12.map((v, i) => v - ema26[i]);
    const dea = calculateEMA(dif, 9);
    const macd = dif.map((v, i) => (v - dea[i]) * 2);
    return { dif, dea, macd };
}

function calculateEMA(data, period) {
    const result = [];
    const mult = 2 / (period + 1);
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
        if (i < period) {
            sum += data[i];
            result.push(sum / (i + 1));
        } else {
            result.push((data[i] - result[i - 1]) * mult + result[i - 1]);
        }
    }
    return result;
}

// 识别分型
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

// 检查2买信号
function checkSecondBuy(data, fractals, idx, lookback = 60) {
    for (let i = fractals.length - 1; i >= 0; i--) {
        const f = fractals[i];
        if (f.type === 'bottom' && f.index < idx && idx - f.index <= lookback) {
            const lowSinceBottom = Math.min(...data.slice(f.index, idx).map(d => d.low));
            if (data[idx].close > f.price && lowSinceBottom > f.price * 0.98) {
                return { bottomPrice: f.price, bottomIdx: f.index };
            }
        }
    }
    return null;
}

// 检查趋势强度 (用于加仓决策)
function checkTrendStrength(data, macdData, idx) {
    if (idx < 20) return false;

    // 价格在均线上方
    const ma20 = data.slice(idx - 20, idx + 1).reduce((s, d) => s + d.close, 0) / 20;
    const aboveMA = data[idx].close > ma20;

    // MACD多头排列
    const bullishMACD = macdData.dif[idx] > macdData.dea[idx] &&
                       macdData.dif[idx] > macdData.dif[idx - 5];

    // 近期低点抬高
    const recentLows = [];
    for (let i = idx - 20; i <= idx; i += 5) {
        const segment = data.slice(Math.max(0, i - 5), i + 5);
        recentLows.push(Math.min(...segment.map(d => d.low)));
    }
    const higherLows = recentLows[recentLows.length - 1] > recentLows[0] * 1.02;

    return aboveMA && bullishMACD && higherLows;
}

function loadCSV(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const lines = content.trim().split('\n');
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const v = lines[i].split(',');
        data.push({
            date: v[0], open: parseFloat(v[1]), high: parseFloat(v[2]),
            low: parseFloat(v[3]), close: parseFloat(v[4]),
            amount: parseFloat(v[5]), volume: parseFloat(v[6])
        });
    }
    return data;
}

// 优化版缠论策略
function backtestEnhanced(data, atr, macdData, fractals) {
    const trades = [];

    let state = {
        inPosition: false,
        entries: [],           // [{price, shares, date, atr}]
        totalShares: 0,
        avgPrice: 0,
        highestPrice: 0,
        highestProfit: 0,
        stopLoss: 0,
        trailingStop: 0,
        entryDate: '',
        entryIndex: 0,
        bottomPrice: 0
    };

    let peakEquity = 0;
    let currentEquity = 100000; // 初始10万

    for (let i = 50; i < data.length; i++) {
        const curr = data[i];
        const currDif = macdData.dif[i] || 0;
        const currDea = macdData.dea[i] || 0;
        const prevDif = macdData.dif[i - 1] || 0;
        const prevDea = macdData.dea[i - 1] || 0;
        const goldenCross = currDif > currDea && prevDif <= prevDea;
        const deathCross = currDif < currDea && prevDif >= prevDea;

        // === 买入信号 ===
        if (!state.inPosition) {
            const secondBuy = checkSecondBuy(data, fractals, i);

            if (secondBuy && goldenCross) {
                const entryATR = atr[i];
                const atrStop = curr.close - entryATR * CONFIG.atrStopMultiplier;

                state = {
                    inPosition: true,
                    entries: [{ price: curr.close, shares: 1, date: curr.date, atr: entryATR }],
                    totalShares: 1,
                    avgPrice: curr.close,
                    highestPrice: curr.close,
                    highestProfit: 0,
                    stopLoss: Math.max(atrStop, secondBuy.bottomPrice * 0.98),
                    trailingStop: 0,
                    entryDate: curr.date,
                    entryIndex: i,
                    bottomPrice: secondBuy.bottomPrice
                };
            }
        }

        // === 持仓中 ===
        if (state.inPosition) {
            const currentProfit = (curr.close - state.avgPrice) / state.avgPrice;
            const unrealizedPnL = (curr.close - state.avgPrice) * state.totalShares;

            // 更新最高价和最高盈利
            if (curr.close > state.highestPrice) {
                state.highestPrice = curr.close;
                state.highestProfit = (state.highestPrice - state.avgPrice) / state.avgPrice;
            }

            // === 加仓逻辑 ===
            if (CONFIG.pyramid.enabled &&
                state.entries.length < CONFIG.pyramid.maxPositions &&
                currentProfit > CONFIG.pyramid.addThreshold) {

                // 检查是否回踩(加仓机会)
                const pullback = (state.highestPrice - curr.close) / state.highestPrice;
                const trendStrong = checkTrendStrength(data, macdData, i);

                if (pullback >= CONFIG.pyramid.pullbackRatio && pullback <= 0.05 && trendStrong) {
                    const addShares = CONFIG.pyramid.addRatio;
                    const oldAvg = state.avgPrice;
                    const newAvg = (state.avgPrice * state.totalShares + curr.close * addShares) /
                                   (state.totalShares + addShares);

                    state.entries.push({
                        price: curr.close,
                        shares: addShares,
                        date: curr.date,
                        atr: atr[i]
                    });
                    state.totalShares += addShares;
                    state.avgPrice = newAvg;

                    // 更新止损
                    const newATRStop = curr.close - atr[i] * CONFIG.atrStopMultiplier;
                    state.stopLoss = Math.max(state.stopLoss, newATRStop);
                }
            }

            // === 移动止损逻辑 ===
            if (CONFIG.trailingStop.enabled && currentProfit > CONFIG.trailingStop.activationProfit) {
                const newTrailStop = state.highestPrice * (1 - CONFIG.trailingStop.trailPercent);
                if (newTrailStop > state.trailingStop) {
                    state.trailingStop = newTrailStop;
                }
            }

            // === 止损/止盈判断 ===
            let exitSignal = false;
            let exitReason = '';
            let exitPrice = curr.close;

            // 1. 移动止损触发
            if (state.trailingStop > 0 && curr.close < state.trailingStop) {
                exitSignal = true;
                exitReason = '移动止损';
                exitPrice = state.trailingStop;
            }
            // 2. ATR/固定止损
            else if (curr.close < state.stopLoss) {
                exitSignal = true;
                exitReason = 'ATR止损';
            }
            // 3. MACD死叉
            else if (deathCross) {
                exitSignal = true;
                exitReason = 'MACD死叉';
            }
            // 4. 顶分型确认(减仓/清仓)
            else if (currentProfit > 0.15) { // 盈利15%以上考虑顶分型
                const recentTop = fractals.filter(f =>
                    f.type === 'top' && f.index >= i - 5 && f.index <= i
                );
                if (recentTop.length > 0) {
                    exitSignal = true;
                    exitReason = '顶分型';
                }
            }

            if (exitSignal) {
                const finalProfit = (exitPrice - state.entries[0].price) / state.entries[0].price * 100;

                trades.push({
                    entryDate: state.entryDate,
                    exitDate: curr.date,
                    entryPrice: state.entries[0].price,
                    exitPrice: exitPrice,
                    profit: finalProfit,
                    holdDays: i - state.entryIndex,
                    reason: exitReason,
                    addCount: state.entries.length - 1,
                    maxProfit: state.highestProfit * 100
                });

                state = {
                    inPosition: false,
                    entries: [],
                    totalShares: 0,
                    avgPrice: 0,
                    highestPrice: 0,
                    highestProfit: 0,
                    stopLoss: 0,
                    trailingStop: 0,
                    entryDate: '',
                    entryIndex: 0,
                    bottomPrice: 0
                };
            }
        }
    }

    return trades;
}

// 原版缠论策略(对比用)
function backtestOriginal(data, macdData, fractals) {
    const trades = [];
    let position = 0, entryPrice = 0, entryDate = '', entryIdx = 0, bottomPrice = 0;

    for (let i = 50; i < data.length; i++) {
        const currDif = macdData.dif[i] || 0, currDea = macdData.dea[i] || 0;
        const prevDif = macdData.dif[i - 1] || 0, prevDea = macdData.dea[i - 1] || 0;
        const goldenCross = currDif > currDea && prevDif <= prevDea;

        if (position === 0) {
            const sb = checkSecondBuy(data, fractals, i);
            if (sb && goldenCross) {
                position = 1; entryPrice = data[i].close; entryDate = data[i].date;
                entryIdx = i; bottomPrice = sb.bottomPrice;
            }
        } else if (position > 0) {
            let sell = false, reason = '';
            if (data[i].close < bottomPrice) { sell = true; reason = '止损'; }
            else if (currDif < currDea && prevDif >= prevDea) { sell = true; reason = 'MACD死叉'; }
            if (sell) {
                trades.push({
                    entryDate, exitDate: data[i].date,
                    profit: (data[i].close / entryPrice - 1) * 100,
                    holdDays: i - entryIdx, reason
                });
                position = 0; bottomPrice = 0;
            }
        }
    }
    return trades;
}

function runComparison() {
    console.log('='.repeat(80));
    console.log('缠论策略优化对比 - 原版 vs 增强版');
    console.log('='.repeat(80));
    console.log('\n优化点:');
    console.log('  1. 动态ATR止损 (替代固定破底分型止损)');
    console.log('  2. 金字塔加仓 (盈利5%后回踩2%加仓50%)');
    console.log('  3. 移动止损 (盈利10%后激活，回撤5%止损)');
    console.log('  4. 顶分型止盈 (盈利15%后遇顶分型出局)');
    console.log('='.repeat(80));

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
        const macdData = calculateMACD(data);
        const fractals = findFractals(data);

        const buyHold = ((data[data.length - 1].close / data[0].close - 1) * 100);
        const origTrades = backtestOriginal(data, macdData, fractals);
        const enhTrades = backtestEnhanced(data, atr, macdData, fractals);

        const origTotal = origTrades.reduce((s, t) => s + t.profit, 0);
        const enhTotal = enhTrades.reduce((s, t) => s + t.profit, 0);

        const origWins = origTrades.filter(t => t.profit > 0).length;
        const enhWins = enhTrades.filter(t => t.profit > 0).length;

        const origAvgProfit = origTrades.length > 0 ? origTotal / origTrades.length : 0;
        const enhAvgProfit = enhTrades.length > 0 ? enhTotal / enhTrades.length : 0;

        // 计算最大回撤
        const origMaxDrawdown = calculateMaxDrawdown(origTrades);
        const enhMaxDrawdown = calculateMaxDrawdown(enhTrades);

        results.push({
            code: stock.code,
            name: stock.name,
            buyHold,
            orig: { total: origTotal, trades: origTrades.length, wins: origWins, avg: origAvgProfit, dd: origMaxDrawdown },
            enh: { total: enhTotal, trades: enhTrades.length, wins: enhWins, avg: enhAvgProfit, dd: enhMaxDrawdown }
        });
    }

    console.log('\n┌────────────┬──────────┬─────────┬─────────┬─────────┬─────────┬─────────┐');
    console.log('│ 股票       │ 买入持有 │ 原版    │ 原版    │ 增强版  │ 增强版  │ 改善    │');
    console.log('│            │ 收益%    │ 收益%   │ 最大回撤│ 收益%   │ 最大回撤│ 收益    │');
    console.log('├────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤');

    for (const r of results) {
        const improvement = ((r.enh.total - r.orig.total) / Math.abs(r.orig.total || 1) * 100).toFixed(1);
        const mark = r.enh.total > r.orig.total ? '+' : '';
        console.log(`│ ${(r.code).padEnd(10)} │ ${r.buyHold.toFixed(5).padStart(7)} │ ${r.orig.total.toFixed(2).padStart(7)} │ ${r.orig.dd.toFixed(1).padStart(7)}% │ ${r.enh.total.toFixed(2).padStart(7)} │ ${r.enh.dd.toFixed(1).padStart(7)}% │ ${mark}${improvement.padStart(6)}% │`);
    }

    console.log('└────────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘');

    // 统计
    const avgOrig = results.reduce((s, r) => s + r.orig.total, 0) / results.length;
    const avgEnh = results.reduce((s, r) => s + r.enh.total, 0) / results.length;
    const beatCount = results.filter(r => r.enh.total > r.orig.total).length;

    console.log(`\n平均收益:`);
    console.log(`  原版: ${avgOrig.toFixed(2)}%`);
    console.log(`  增强版: ${avgEnh.toFixed(2)}% (改善 ${(avgEnh - avgOrig).toFixed(2)}%)`);
    console.log(`\n超越原版: ${beatCount}/${results.length} 只股票`);

    // 详细交易示例
    console.log('\n' + '='.repeat(80));
    console.log('增强版策略交易示例 (比亚迪 sz002594)');
    console.log('='.repeat(80));

    const bydCsv = path.join(__dirname, 'test_output', 'sz002594.day.csv');
    if (fs.existsSync(bydCsv)) {
        const data = loadCSV(bydCsv);
        const atr = calculateATR(data);
        const macdData = calculateMACD(data);
        const fractals = findFractals(data);
        const trades = backtestEnhanced(data, atr, macdData, fractals);

        if (trades.length > 0) {
            console.log('\n交易记录:');
            trades.forEach((t, i) => {
                const profitStr = t.profit >= 0 ? `+${t.profit.toFixed(2)}%` : `${t.profit.toFixed(2)}%`;
                const addStr = t.addCount > 0 ? ` [加仓${t.addCount}次]` : '';
                console.log(`  ${i + 1}. ${t.entryDate} -> ${t.exitDate} (${t.holdDays}天) ${profitStr} ${t.reason}${addStr}`);
                if (t.maxProfit > 0) {
                    console.log(`     最高浮盈: +${t.maxProfit.toFixed(2)}%`);
                }
            });
        }
    }

    console.log('\n' + '='.repeat(80));
}

function calculateMaxDrawdown(trades) {
    if (trades.length === 0) return 0;

    let maxDrawdown = 0;
    let peak = 0;
    let cumulative = 0;

    for (const t of trades) {
        cumulative += t.profit;
        if (cumulative > peak) peak = cumulative;
        const drawdown = peak - cumulative;
        if (drawdown > maxDrawdown) maxDrawdown = drawdown;
    }

    return maxDrawdown;
}

runComparison();
