#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// 计算MACD
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

    let sum = 0;
    for (let i = 0; i < period && i < data.length; i++) {
        sum += data[i];
        result.push(sum / (i + 1));
    }

    for (let i = period; i < data.length; i++) {
        const ema = (data[i] - result[i - 1]) * multiplier + result[i - 1];
        result.push(ema);
    }

    return result;
}

// 识别分型
function findFractals(data) {
    const fractals = [];

    for (let i = 1; i < data.length - 1; i++) {
        const prev = data[i - 1];
        const curr = data[i];
        const next = data[i + 1];

        // 顶分型
        if (curr.high > prev.high && curr.high > next.high &&
            curr.low > prev.low && curr.low > next.low) {
            fractals.push({ index: i, type: 'top', date: curr.date, price: curr.high });
        }
        // 底分型
        else if (curr.low < prev.low && curr.low < next.low &&
                 curr.high < prev.high && curr.high < next.high) {
            fractals.push({ index: i, type: 'bottom', date: curr.date, price: curr.low });
        }
    }

    return fractals;
}

// 转换为周线
function resampleToWeekly(dailyData) {
    const weeklyData = [];
    let currentWeek = null;
    let weekCount = 0;

    for (let i = 0; i < dailyData.length; i++) {
        const d = dailyData[i];

        if (i % 5 === 0) { // 每5天作为一周(简化)
            if (currentWeek) {
                weeklyData.push(currentWeek);
            }
            currentWeek = {
                date: d.date,
                index: weekCount++,
                open: d.open,
                high: d.high,
                low: d.low,
                close: d.close,
                volume: d.volume
            };
        } else if (currentWeek) {
            currentWeek.high = Math.max(currentWeek.high, d.high);
            currentWeek.low = Math.min(currentWeek.low, d.low);
            currentWeek.close = d.close;
            currentWeek.volume += d.volume;
        }
    }

    if (currentWeek) {
        weeklyData.push(currentWeek);
    }

    return weeklyData;
}

// 检查底分型后回踩 (2买信号)
function checkSecondBuy(dailyData, dailyFractals, dailyIndex, lookbackDays = 60) {
    // 找到最近的底分型
    let recentBottom = null;
    for (let i = dailyFractals.length - 1; i >= 0; i--) {
        const f = dailyFractals[i];
        if (f.type === 'bottom' && f.index < dailyIndex) {
            if (dailyIndex - f.index <= lookbackDays) {
                recentBottom = f;
                break;
            }
        }
    }

    if (!recentBottom) return null;

    // 检查是否形成回踩不破底分型低点
    const current = dailyData[dailyIndex];
    const lowSinceBottom = Math.min(
        ...dailyData.slice(recentBottom.index, dailyIndex).map(d => d.low)
    );

    // 回踩但未破前低
    if (current.close > recentBottom.price && lowSinceBottom > recentBottom.price * 0.98) {
        return {
            bottomDate: recentBottom.date,
            bottomPrice: recentBottom.price,
            currentPrice: current.close
        };
    }

    return null;
}

// 检查MACD顶背离
function checkTopDivergence(data, macdData, index, lookback = 30) {
    if (index < lookback + 5) return false;

    const recentHighs = [];
    for (let i = index - lookback; i <= index; i++) {
        if (i > 0 && data[i].high > data[i-1].high && data[i].high > data[i+1]?.high) {
            recentHighs.push({ index: i, price: data[i].high, macd: macdData.macd[i] });
        }
    }

    if (recentHighs.length < 2) return false;

    // 检查最新高点
    const latest = recentHighs[recentHighs.length - 1];
    const prev = recentHighs[recentHighs.length - 2];

    return latest.price > prev.price && latest.macd < prev.macd;
}

function loadCSV(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const lines = content.trim().split('\n');

    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        data.push({
            date: values[0],
            open: parseFloat(values[1]),
            high: parseFloat(values[2]),
            low: parseFloat(values[3]),
            close: parseFloat(values[4]),
            amount: parseFloat(values[5]),
            volume: parseFloat(values[6])
        });
    }

    return data;
}

function runBacktest() {
    console.log('='.repeat(65));
    console.log('002600 驰宏锌锗 缠论策略回测 v2.0');
    console.log('='.repeat(65));
    console.log('策略: 日线底分型后2买, MACD确认, 跌破底分型止损');
    console.log('       MACD顶背离减仓50%, MACD死叉清仓');
    console.log('='.repeat(65));

    const csvFile = path.join(__dirname, 'test_output', 'sz002600.day.csv');
    const dailyData = loadCSV(csvFile);

    console.log(`\n数据概览:`);
    console.log(`  日期范围: ${dailyData[0].date} ~ ${dailyData[dailyData.length - 1].date}`);
    console.log(`  K线数量: ${dailyData.length} 条`);

    const buyHoldReturn = ((dailyData[dailyData.length - 1].close / dailyData[0].close - 1) * 100).toFixed(2);
    console.log(`  买入持有收益: ${buyHoldReturn}%`);

    // 计算MACD
    const dailyMACD = calculateMACD(dailyData);

    // 识别分型
    const dailyFractals = findFractals(dailyData);
    const bottomFractals = dailyFractals.filter(f => f.type === 'bottom');
    const topFractals = dailyFractals.filter(f => f.type === 'top');

    console.log(`\n缠论结构:`);
    console.log(`  底分型: ${bottomFractals.length} 个`);
    console.log(`  顶分型: ${topFractals.length} 个`);

    console.log('\n' + '='.repeat(65));
    console.log('交易记录');
    console.log('='.repeat(65));

    const trades = [];
    let position = 0; // 0=空, 0.5=半, 1=满
    let entryPrice = 0;
    let entryDate = '';
    let entryIndex = 0;
    let bottomPrice = 0; // 止损参考价
    let partialExitPrice = 0; // 减仓价格

    for (let i = 50; i < dailyData.length; i++) {
        const curr = dailyData[i];
        const currDif = dailyMACD.dif[i] || 0;
        const currDea = dailyMACD.dea[i] || 0;
        const prevDif = dailyMACD.dif[i - 1] || 0;
        const prevDea = dailyMACD.dea[i - 1] || 0;
        const goldenCross = currDif > currDea && prevDif <= prevDea;
        const deathCross = currDif < currDea && prevDif >= prevDea;

        // === 买入信号 (2买) ===
        if (position === 0) {
            const secondBuy = checkSecondBuy(dailyData, dailyFractals, i);

            if (secondBuy && goldenCross) {
                position = 1;
                entryPrice = curr.close;
                entryDate = curr.date;
                entryIndex = i;
                bottomPrice = secondBuy.bottomPrice;
                partialExitPrice = curr.close;

                console.log(`[买入] ${curr.date} @ ${entryPrice.toFixed(2)} (2买, 底分型${secondBuy.bottomDate} ${bottomPrice.toFixed(2)}, MACD金叉)`);
            }
        }

        // === 卖出信号 ===
        if (position > 0) {
            let actionTaken = false;

            // 1. 止损: 跌破底分型价格
            if (curr.close < bottomPrice) {
                const profit = ((curr.close - entryPrice) / entryPrice * 100);
                trades.push({
                    entryDate, entryPrice, exitDate: curr.date, exitPrice: curr.close,
                    profit, reason: '止损', holdDays: i - entryIndex
                });
                console.log(`[卖出] ${curr.date} @ ${curr.close.toFixed(2)} (止损破${bottomPrice.toFixed(2)}) ${profit.toFixed(2)}%`);
                position = 0;
                entryPrice = 0;
                bottomPrice = 0;
                actionTaken = true;
            }
            // 2. MACD顶背离 - 减仓50%
            else if (position === 1 && checkTopDivergence(dailyData, dailyMACD, i)) {
                const unrealizedProfit = ((curr.close - entryPrice) / entryPrice * 100);
                console.log(`[减仓50%%] ${curr.date} @ ${curr.close.toFixed(2)} (MACD顶背离, 浮盈${unrealizedProfit.toFixed(2)}%)`);
                position = 0.5;
                partialExitPrice = curr.close;
                actionTaken = true;
            }
            // 3. MACD死叉 - 清仓
            else if (deathCross) {
                const profit = ((curr.close - partialExitPrice) / partialExitPrice * 100);
                trades.push({
                    entryDate, entryPrice, exitDate: curr.date, exitPrice: curr.close,
                    profit, reason: position === 1 ? 'MACD死叉' : 'MACD死叉(半仓)',
                    holdDays: i - entryIndex
                });
                console.log(`[清仓] ${curr.date} @ ${curr.close.toFixed(2)} (MACD死叉) ${profit.toFixed(2)}%`);
                position = 0;
                entryPrice = 0;
                bottomPrice = 0;
                partialExitPrice = 0;
                actionTaken = true;
            }
        }
    }

    // 统计
    console.log('\n' + '='.repeat(65));
    console.log('回测结果');
    console.log('='.repeat(65));

    if (trades.length > 0) {
        const winning = trades.filter(t => t.profit > 0);
        const losing = trades.filter(t => t.profit <= 0);

        const totalProfit = trades.reduce((sum, t) => sum + t.profit, 0);
        const avgProfit = totalProfit / trades.length;

        console.log(`  交易次数: ${trades.length}`);
        console.log(`  盈利: ${winning.length} | 亏损: ${losing.length}`);
        console.log(`  胜率: ${(winning.length / trades.length * 100).toFixed(1)}%`);
        console.log(`  总收益: ${totalProfit.toFixed(2)}%`);
        console.log(`  平均收益: ${avgProfit.toFixed(2)}%`);

        if (winning.length > 0) {
            const avgWin = winning.reduce((s, t) => s + t.profit, 0) / winning.length;
            const maxWin = Math.max(...winning.map(t => t.profit));
            console.log(`  平均盈利: ${avgWin.toFixed(2)}% | 最大: ${maxWin.toFixed(2)}%`);
        }

        if (losing.length > 0) {
            const avgLoss = losing.reduce((s, t) => s + t.profit, 0) / losing.length;
            const maxLoss = Math.min(...losing.map(t => t.profit));
            console.log(`  平均亏损: ${avgLoss.toFixed(2)}% | 最大: ${maxLoss.toFixed(2)}%`);
        }

        if (winning.length > 0 && losing.length > 0) {
            const totalWin = winning.reduce((s, t) => s + t.profit, 0);
            const totalLoss = Math.abs(losing.reduce((s, t) => s + t.profit, 0));
            console.log(`  盈亏比: ${(totalWin / totalLoss).toFixed(2)}`);
        }

        const avgHold = trades.reduce((s, t) => s + t.holdDays, 0) / trades.length;
        console.log(`  平均持仓: ${avgHold.toFixed(0)}天`);

        // 显示各笔交易
        console.log('\n  各笔交易详情:');
        trades.forEach((t, idx) => {
            const profitStr = t.profit >= 0 ? `+${t.profit.toFixed(2)}%` : `${t.profit.toFixed(2)}%`;
            console.log(`    ${idx + 1}. ${t.entryDate} -> ${t.exitDate} (${t.holdDays}天) ${profitStr} [${t.reason}]`);
        });
    } else {
        console.log('  无交易信号');
    }

    console.log(`\n  买入持有: ${buyHoldReturn}%`);
    console.log(`  策略收益: ${trades.reduce((s, t) => s + t.profit, 0).toFixed(2)}%`);

    const excessReturn = (parseFloat(trades.reduce((s, t) => s + t.profit, 0).toFixed(2)) - parseFloat(buyHoldReturn)).toFixed(2);
    console.log(`  超额收益: ${excessReturn}%`);

    console.log('\n回测完成!');
}

runBacktest();
