#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const CONFIG = {
    stopLoss: {
        mode: 'hybrid',
        atrPeriod: 14,
        atrMultiplier: 1.5,
        trailingEnabled: true,
        trailingActivation: 0.10,
        trailingDistance: 0.05
    }
};

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
                return { bottomPrice: f.price, bottomIdx: f.index, bottomDate: f.date };
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

function backtestDetailed(stockCode, stockName, data, atr, macdData, fractals) {
    const trades = [];
    let state = {
        inPosition: false,
        entryPrice: 0,
        entryDate: '',
        entryIndex: 0,
        bottomPrice: 0,
        bottomDate: '',
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
                    bottomDate: secondBuy.bottomDate,
                    atrStop: atrStop,
                    highestPrice: curr.close,
                    trailingStop: 0
                };
            }
        }

        if (state.inPosition) {
            const profit = (curr.close - state.entryPrice) / state.entryPrice;
            if (curr.close > state.highestPrice) {
                state.highestPrice = curr.close;
            }

            if (CONFIG.stopLoss.trailingEnabled && profit >= CONFIG.stopLoss.trailingActivation) {
                state.trailingStop = state.highestPrice * (1 - CONFIG.stopLoss.trailingDistance);
            }

            let exit = false, reason = '', exitPrice = curr.close;

            if (state.trailingStop > 0 && curr.close < state.trailingStop) {
                exit = true; reason = '移动止损'; exitPrice = state.trailingStop;
            } else if (curr.close < state.bottomPrice || curr.close < state.atrStop) {
                exit = true; reason = '止损';
            } else if (deathCross) {
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
                    maxProfit: (state.highestPrice - state.entryPrice) / state.entryPrice * 100,
                    bottomDate: state.bottomDate
                });

                state = {
                    inPosition: false, entryPrice: 0, entryDate: '',
                    entryIndex: 0, bottomPrice: 0, bottomDate: '',
                    atrStop: 0, highestPrice: 0, trailingStop: 0
                };
            }
        }
    }

    return { trades, stockCode, stockName };
}

function printTradeDetail(result, buyHold) {
    const { trades, stockCode, stockName } = result;

    console.log('\n' + '═'.repeat(90));
    console.log(` ${stockCode} ${stockName} - 详细交易记录`);
    console.log('═'.repeat(90));

    const totalProfit = trades.reduce((s, t) => s + t.profit, 0);
    const winningTrades = trades.filter(t => t.profit > 0);
    const losingTrades = trades.filter(t => t.profit <= 0);

    console.log(`\n买入持有收益: ${buyHold.toFixed(2)}%`);
    console.log(`策略收益: ${totalProfit.toFixed(2)}% (${totalProfit > buyHold ? '+' : ''}${(totalProfit - buyHold).toFixed(2)}%)`);
    console.log(`交易次数: ${trades.length}`);
    console.log(`盈利: ${winningTrades.length} | 亏损: ${losingTrades.length} | 胜率: ${(winningTrades.length/trades.length*100).toFixed(0)}%`);

    if (winningTrades.length > 0) {
        const avgWin = winningTrades.reduce((s, t) => s + t.profit, 0) / winningTrades.length;
        const maxWin = Math.max(...winningTrades.map(t => t.profit));
        console.log(`平均盈利: ${avgWin.toFixed(2)}% | 最大盈利: ${maxWin.toFixed(2)}%`);
    }

    if (losingTrades.length > 0) {
        const avgLoss = losingTrades.reduce((s, t) => s + t.profit, 0) / losingTrades.length;
        const maxLoss = Math.min(...losingTrades.map(t => t.profit));
        console.log(`平均亏损: ${avgLoss.toFixed(2)}% | 最大亏损: ${maxLoss.toFixed(2)}%`);
    }

    console.log('\n┌──────┬─────────────┬─────────────┬─────────┬─────────┬─────────┬─────────┬─────────┐');
    console.log('│ No.  │ 入场日期    │ 出场日期    │ 入场价  │ 出场价  │ 收益%   │ 持仓天数│ 原因    │');
    console.log('├──────┼─────────────┼─────────────┼─────────┼─────────┼─────────┼─────────┼─────────┤');

    trades.forEach((t, i) => {
        const profitStr = t.profit >= 0 ? `\x1b[32m+${t.profit.toFixed(2)}%\x1b[0m` : `\x1b[31m${t.profit.toFixed(2)}%\x1b[0m`;
        const maxProfitStr = t.maxProfit > 0 ? ` (最高+${t.maxProfit.toFixed(1)}%)` : '';
        console.log(`│ ${String(i + 1).padStart(4)} │ ${t.entryDate} │ ${t.exitDate} │ ${t.entryPrice.toFixed(2).padStart(7)} │ ${t.exitPrice.toFixed(2).padStart(7)} │ ${profitStr.padStart(7)} │ ${String(t.holdDays).padStart(7)} │ ${t.reason.padEnd(6)} │${maxProfitStr}`);
    });

    console.log('└──────┴─────────────┴─────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘');

    // 出场原因统计
    const exitReasons = {};
    trades.forEach(t => {
        exitReasons[t.reason] = (exitReasons[t.reason] || 0) + 1;
    });
    console.log('\n出场原因统计:');
    Object.entries(exitReasons).forEach(([reason, count]) => {
        console.log(`  ${reason}: ${count}次`);
    });
}

function runAllBacktests() {
    console.log('='.repeat(90));
    console.log(' 极简优化缠论策略 - 详细回测报告');
    console.log('='.repeat(90));

    const stocks = [
        { code: 'sz002600', name: '驰宏锌锗' },
        { code: 'sh600519', name: '贵州茅台' },
        { code: 'sz000858', name: '五粮液' },
        { code: 'sh600036', name: '招商银行' },
        { code: 'sz002594', name: '比亚迪' },
        { code: 'sz300750', name: '宁德时代' }
    ];

    const allResults = [];

    for (const stock of stocks) {
        const csvFile = path.join(__dirname, 'test_output', `${stock.code}.day.csv`);
        if (!fs.existsSync(csvFile)) {
            console.log(`\n[跳过] ${stock.code} ${stock.name} - 文件不存在`);
            continue;
        }

        const data = loadCSV(csvFile);
        const atr = calculateATR(data);
        const macdData = calculateMACD(data);
        const fractals = findFractals(data);

        const buyHold = ((data[data.length - 1].close / data[0].close - 1) * 100);
        const result = backtestDetailed(stock.code, stock.name, data, atr, macdData, fractals);

        printTradeDetail(result, buyHold);
        allResults.push({ ...result, buyHold });
    }

    // 汇总报告
    console.log('\n' + '='.repeat(90));
    console.log(' 汇总报告');
    console.log('='.repeat(90));

    const avgProfit = allResults.reduce((s, r) => s + r.trades.reduce((a, t) => a + t.profit, 0), 0) / allResults.length;
    const avgBuyHold = allResults.reduce((s, r) => s + r.buyHold, 0) / allResults.length;
    const beatCount = allResults.filter(r => r.trades.reduce((a, t) => a + t.profit, 0) > r.buyHold).length;

    console.log(`\n平均收益 (策略): ${avgProfit.toFixed(2)}%`);
    console.log(`平均收益 (买入持有): ${avgBuyHold.toFixed(2)}%`);
    console.log(`超越买入持有: ${beatCount}/${allResults.length} 只股票`);
    console.log(`超额收益: ${(avgProfit - avgBuyHold).toFixed(2)}%`);

    console.log('\n' + '='.repeat(90));
}

runAllBacktests();
