#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// Simple MACD calculation
function calculateMACD(data, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
    const closes = data.map(d => d.close);
    const emaFast = calculateEMA(closes, fastPeriod);
    const emaSlow = calculateEMA(closes, slowPeriod);

    const dif = emaFast.map((f, i) => f - emaSlow[i]);
    const dea = calculateEMA(dif, signalPeriod);
    const macd = dif.map((d, i) => (d - dea[i]) * 2);

    return { dif, dea, macd };
}

function calculateEMA(data, period) {
    const result = [];
    const multiplier = 2 / (period + 1);

    // Start with SMA for first period
    let sum = 0;
    for (let i = 0; i < period && i < data.length; i++) {
        sum += data[i];
        result.push(sum / (i + 1));
    }

    // EMA for rest
    for (let i = period; i < data.length; i++) {
        const ema = (data[i] - result[i - 1]) * multiplier + result[i - 1];
        result.push(ema);
    }

    return result;
}

// Find fractals (simplified - 3-kline pattern)
function findFractals(data) {
    const fractals = [];

    for (let i = 1; i < data.length - 1; i++) {
        const prev = data[i - 1];
        const curr = data[i];
        const next = data[i + 1];

        // Top fractal: middle has highest high and highest low
        if (curr.high > prev.high && curr.high > next.high &&
            curr.low > prev.low && curr.low > next.low) {
            fractals.push({ index: i, type: 'top', date: curr.date });
        }
        // Bottom fractal: middle has lowest low and lowest high
        else if (curr.low < prev.low && curr.low < next.low &&
                 curr.high < prev.high && curr.high < next.high) {
            fractals.push({ index: i, type: 'bottom', date: curr.date });
        }
    }

    return fractals;
}

// Load TDX CSV data
function loadCSV(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const lines = content.trim().split('\n');
    const headers = lines[0].split(',');

    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        const obj = {};
        headers.forEach((h, idx) => obj[h.trim()] = parseFloat(values[idx]));
        obj.date = values[0]; // Keep as string for display
        obj.open = parseFloat(values[1]);
        obj.high = parseFloat(values[2]);
        obj.low = parseFloat(values[3]);
        obj.close = parseFloat(values[4]);
        obj.amount = parseFloat(values[5]);
        obj.volume = parseFloat(values[6]);
        data.push(obj);
    }

    return data;
}

// Run backtest
function runBacktest() {
    console.log('='.repeat(60));
    console.log('002600 驰宏锌锗 回测报告 (通达信数据)');
    console.log('='.repeat(60));

    const csvFile = path.join(__dirname, 'test_output', 'sz002600.day.csv');

    if (!fs.existsSync(csvFile)) {
        console.log(`错误: 数据文件不存在: ${csvFile}`);
        return;
    }

    const data = loadCSV(csvFile);

    console.log(`\n数据概览:`);
    console.log(`  日期范围: ${data[0].date} ~ ${data[data.length - 1].date}`);
    console.log(`  K线数量: ${data.length} 条`);
    console.log(`  起始价格: ${data[0].close.toFixed(2)}`);
    console.log(`  最新价格: ${data[data.length - 1].close.toFixed(2)}`);

    const buyHoldReturn = ((data[data.length - 1].close / data[0].close - 1) * 100).toFixed(2);
    console.log(`  期间涨跌: ${buyHoldReturn}%`);

    // Calculate MACD
    const macdData = calculateMACD(data);

    // Find fractals
    const fractals = findFractals(data);
    const topFractals = fractals.filter(f => f.type === 'top');
    const bottomFractals = fractals.filter(f => f.type === 'bottom');

    console.log(`\n缠论结构识别:`);
    console.log(`  顶分型: ${topFractals.length} 个`);
    console.log(`  底分型: ${bottomFractals.length} 个`);

    console.log('\n' + '='.repeat(60));
    console.log('策略回测 (简化版: MACD金叉买入, 死叉卖出)');
    console.log('='.repeat(60));

    const trades = [];
    let position = false;
    let entryPrice = 0;
    let entryDate = '';
    let entryIndex = 0;

    for (let i = 50; i < data.length; i++) {
        const curr = data[i];
        const prev = data[i - 1];

        const currDif = macdData.dif[i] || 0;
        const currDea = macdData.dea[i] || 0;
        const prevDif = macdData.dif[i - 1] || 0;
        const prevDea = macdData.dea[i - 1] || 0;

        // Buy signal: MACD golden cross
        if (!position && currDif > currDea && prevDif <= prevDea) {
            position = true;
            entryPrice = curr.close;
            entryDate = curr.date;
            entryIndex = i;
            console.log(`买入: ${curr.date} @ ${entryPrice.toFixed(2)}`);
        }
        // Sell signal: MACD death cross or stop loss
        else if (position) {
            let sellSignal = false;
            let reason = '';

            // Stop loss: -5%
            if (curr.close < entryPrice * 0.95) {
                sellSignal = true;
                reason = '止损(-5%)';
            }
            // MACD death cross
            else if (currDif < currDea && prevDif >= prevDea) {
                sellSignal = true;
                reason = 'MACD死叉';
            }

            if (sellSignal) {
                const profit = ((curr.close / entryPrice - 1) * 100).toFixed(2);
                const holdDays = i - entryIndex;
                trades.push({
                    entryDate,
                    entryPrice,
                    exitDate: curr.date,
                    exitPrice: curr.close,
                    profit: parseFloat(profit),
                    reason,
                    holdDays
                });
                console.log(`卖出: ${curr.date} @ ${curr.close.toFixed(2)} (${reason}) 收益: ${profit}% (${holdDays}天)`);
                position = false;
                entryPrice = 0;
            }
        }
    }

    // Statistics
    console.log('\n' + '='.repeat(60));
    console.log('回测结果汇总');
    console.log('='.repeat(60));

    if (trades.length > 0) {
        const winningTrades = trades.filter(t => t.profit > 0);
        const losingTrades = trades.filter(t => t.profit <= 0);

        const totalProfit = trades.reduce((sum, t) => sum + t.profit, 0);
        const avgProfit = totalProfit / trades.length;

        console.log(`  总交易次数: ${trades.length}`);
        console.log(`  盈利次数: ${winningTrades.length}`);
        console.log(`  亏损次数: ${losingTrades.length}`);
        console.log(`  胜率: ${(winningTrades.length / trades.length * 100).toFixed(2)}%`);
        console.log(`  总收益: ${totalProfit.toFixed(2)}%`);
        console.log(`  平均收益: ${avgProfit.toFixed(2)}%`);

        if (winningTrades.length > 0) {
            const avgWin = winningTrades.reduce((sum, t) => sum + t.profit, 0) / winningTrades.length;
            const maxWin = Math.max(...winningTrades.map(t => t.profit));
            console.log(`  平均盈利: ${avgWin.toFixed(2)}%`);
            console.log(`  最大盈利: ${maxWin.toFixed(2)}%`);
        }

        if (losingTrades.length > 0) {
            const avgLoss = losingTrades.reduce((sum, t) => sum + t.profit, 0) / losingTrades.length;
            const maxLoss = Math.min(...losingTrades.map(t => t.profit));
            console.log(`  平均亏损: ${avgLoss.toFixed(2)}%`);
            console.log(`  最大亏损: ${maxLoss.toFixed(2)}%`);
        }

        if (winningTrades.length > 0 && losingTrades.length > 0) {
            const totalWin = winningTrades.reduce((sum, t) => sum + t.profit, 0);
            const totalLoss = Math.abs(losingTrades.reduce((sum, t) => sum + t.profit, 0));
            console.log(`  盈亏比: ${(totalWin / totalLoss).toFixed(2)}`);
        }

        const avgHoldDays = trades.reduce((sum, t) => sum + t.holdDays, 0) / trades.length;
        console.log(`  平均持仓: ${avgHoldDays.toFixed(0)} 天`);
    } else {
        console.log('  未产生交易信号');
    }

    console.log(`\n  买入持有收益: ${buyHoldReturn}%`);
    console.log('\n回测完成!');
}

runBacktest();
