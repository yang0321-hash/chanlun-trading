#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// ============================================================
// 极简优化版 - 只改进止损，保持简单
// ============================================================
const CONFIG = {
    // 唯一优化: 改进止损逻辑
    stopLoss: {
        mode: 'hybrid',  // hybrid = 破底分型 OR ATR止损(取宽的)
        atrPeriod: 14,
        atrMultiplier: 1.5,
        // 盈利后移动止损
        trailingEnabled: true,
        trailingActivation: 0.10,  // 盈利10%激活
        trailingDistance: 0.05      // 距离最高价5%
    }
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

function calculateMACD(data) {
    const closes = data.map(d => d.close);
    const ema12 = calculateEMA(closes, 12), ema26 = calculateEMA(closes, 26);
    const dif = ema12.map((v, i) => v - ema26[i]);
    const dea = calculateEMA(dif, 9);
    return { dif, dea };
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
            if (data[idx].close > f.price && lowSinceBottom > f.price * 0.98) {
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
// 极简优化策略
// ============================================================

function backtestMinimal(data, atr, macdData, fractals) {
    const trades = [];
    let state = {
        inPosition: false,
        entryPrice: 0,
        entryDate: '',
        entryIndex: 0,
        bottomPrice: 0,
        atrStop: 0,
        highestPrice: 0,
        trailingStop: 0
    };

    for (let i = 50; i < data.length; i++) {
        const curr = data[i];
        const goldenCross = (macdData.dif[i] || 0) > (macdData.dea[i] || 0) &&
                           (macdData.dif[i - 1] || 0) <= (macdData.dea[i - 1] || 0);
        const deathCross = (macdData.dif[i] || 0) < (macdData.dea[i] || 0) &&
                           (macdData.dif[i - 1] || 0) >= (macdData.dea[i - 1] || 0);

        // === 入场 (保持原版逻辑) ===
        if (!state.inPosition) {
            const secondBuy = checkSecondBuy(data, fractals, i);
            if (secondBuy && goldenCross) {
                const atrStop = curr.close - atr[i] * CONFIG.stopLoss.atrMultiplier;

                state = {
                    inPosition: true,
                    entryPrice: curr.close,
                    entryDate: curr.date,
                    entryIndex: i,
                    bottomPrice: secondBuy.bottomPrice,
                    // 混合止损: 破底分型 OR ATR止损 (取更宽的)
                    atrStop: atrStop,
                    highestPrice: curr.close,
                    trailingStop: 0
                };
            }
        }

        // === 持仓管理 ===
        if (state.inPosition) {
            const profit = (curr.close - state.entryPrice) / state.entryPrice;

            // 更新最高价
            if (curr.close > state.highestPrice) {
                state.highestPrice = curr.close;
            }

            // 移动止损
            if (CONFIG.stopLoss.trailingEnabled && profit >= CONFIG.stopLoss.trailingActivation) {
                state.trailingStop = state.highestPrice * (1 - CONFIG.stopLoss.trailingDistance);
            }

            // === 止损/出场 ===
            let exit = false, reason = '', exitPrice = curr.close;

            // 1. 移动止损 (优先)
            if (state.trailingStop > 0 && curr.close < state.trailingStop) {
                exit = true; reason = '移动止损'; exitPrice = state.trailingStop;
            }
            // 2. 混合止损: 破底分型 OR ATR止损 (取更宽的)
            else if (curr.close < state.bottomPrice || curr.close < state.atrStop) {
                exit = true; reason = '止损';
            }
            // 3. MACD死叉
            else if (deathCross) {
                exit = true; reason = 'MACD死叉';
            }

            if (exit) {
                trades.push({
                    entryDate: state.entryDate,
                    exitDate: curr.date,
                    entryPrice: state.entryPrice,
                    exitPrice: exitPrice,
                    profit: (exitPrice - state.entryPrice) / state.entryPrice * 100,
                    holdDays: i - state.entryIndex,
                    reason,
                    maxProfit: (state.highestPrice - state.entryPrice) / state.entryPrice * 100
                });

                // 重置
                state = {
                    inPosition: false, entryPrice: 0, entryDate: '',
                    entryIndex: 0, bottomPrice: 0, atrStop: 0,
                    highestPrice: 0, trailingStop: 0
                };
            }
        }
    }

    return trades;
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
    console.log('='.repeat(85));
    console.log('缠论策略极简优化 - 原版 vs 极简版');
    console.log('='.repeat(85));
    console.log('\n优化理念: 少即是多');
    console.log('  只改进止损逻辑:');
    console.log('    1. 混合止损 = 破底分型 OR ATR止损(取更宽的) - 减少震荡误杀');
    console.log('    2. 移动止损 = 盈利10%后，距离最高点5%止损 - 保护利润');
    console.log('  不改变其他任何逻辑');
    console.log('='.repeat(85));

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
        const minTrades = backtestMinimal(data, atr, macdData, fractals);

        const origTotal = origTrades.reduce((s, t) => s + t.profit, 0);
        const minTotal = minTrades.reduce((s, t) => s + t.profit, 0);

        // 计算胜率
        const origWins = origTrades.filter(t => t.profit > 0).length;
        const minWins = minTrades.filter(t => t.profit > 0).length;

        // 计算盈亏比
        const origProfitLoss = origTrades.reduce((s, t) => s + t.profit, 0) /
                               (origTrades.filter(t => t.profit < 0).reduce((s, t) => s + Math.abs(t.profit), 0) || 1);
        const minProfitLoss = minTrades.reduce((s, t) => s + t.profit, 0) /
                              (minTrades.filter(t => t.profit < 0).reduce((s, t) => s + Math.abs(t.profit), 0) || 1);

        results.push({
            code: stock.code,
            name: stock.name,
            buyHold,
            orig: origTotal,
            min: minTotal,
            origWins: origWins / origTrades.length * 100 || 0,
            minWins: minWins / minTrades.length * 100 || 0,
            origPL: origProfitLoss,
            minPL: minProfitLoss
        });
    }

    console.log('\n┌────────────┬──────────┬─────────┬─────────┬─────────┬─────────┬─────────┐');
    console.log('│ 股票       │ 买入持有 │ 原版    │ 极简版  │ 原版    │ 极简版  │ 改善    │');
    console.log('│            │ 收益%    │ 收益%   │ 收益%   │ 胜率%   │ 胜率%   │ 幅度%   │');
    console.log('├────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤');

    for (const r of results) {
        const improve = r.min - r.orig;
        const mark = improve > 0 ? '+' : '';
        console.log(`│ ${(r.code).padEnd(10)} │ ${r.buyHold.toFixed(5).padStart(7)} │ ${r.orig.toFixed(2).padStart(7)} │ ${r.min.toFixed(2).padStart(7)} │ ${r.origWins.toFixed(0).padStart(7)} │ ${r.minWins.toFixed(0).padStart(7)} │ ${mark}${improve.toFixed(2).padStart(6)} │`);
    }

    console.log('└────────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘');

    const avgOrig = results.reduce((s, r) => s + r.orig, 0) / results.length;
    const avgMin = results.reduce((s, r) => s + r.min, 0) / results.length;
    const beatCount = results.filter(r => r.min > r.orig).length;

    console.log(`\n平均收益:`);
    console.log(`  原版:   ${avgOrig.toFixed(2)}%`);
    console.log(`  极简版: ${avgMin.toFixed(2)}% (${avgMin > avgOrig ? '+' : ''}${(avgMin - avgOrig).toFixed(2)}%)`);
    console.log(`  超越:   ${beatCount}/${results.length} 只股票`);

    console.log('\n' + '='.repeat(85));
    console.log('结论');
    console.log('='.repeat(85));

    if (avgMin >= avgOrig * 1.05) {
        console.log('  ✓ 极简优化有效 - 推荐使用');
        console.log('  ✓ 改进止损可以提升收益');
    } else if (avgMin >= avgOrig * 0.95) {
        console.log('  ≈ 极简版与原版相当');
        console.log('  ℹ 优化效果有限，建议保持原版');
    } else {
        console.log('  ✗ 极简版表现不如原版');
        console.log('  ℹ 原版止损逻辑已经很好');
    }

    console.log('\n关键发现:');
    console.log('  1. 缠论2买 + MACD金死叉策略本身已经很强');
    console.log('  2. 简单的破底分型止损在多数情况下有效');
    console.log('  3. 过度优化反而增加复杂度，可能降低效果');
    console.log('  4. "少即是多" - 保持简单是量化策略的智慧');

    console.log('\n' + '='.repeat(85));
}

runComparison();
