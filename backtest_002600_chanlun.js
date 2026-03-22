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

// 识别分型 (3K模式)
function findFractals(data) {
    const fractals = [];

    for (let i = 1; i < data.length - 1; i++) {
        const prev = data[i - 1];
        const curr = data[i];
        const next = data[i + 1];

        // 顶分型: 中间K线高点最高，低点也最高
        if (curr.high > prev.high && curr.high > next.high &&
            curr.low > prev.low && curr.low > next.low) {
            fractals.push({ index: i, type: 'top', date: curr.date, price: curr.high });
        }
        // 底分型: 中间K线低点最低，高点也最低
        else if (curr.low < prev.low && curr.low < next.low &&
                 curr.high < prev.high && curr.high < next.high) {
            fractals.push({ index: i, type: 'bottom', date: curr.date, price: curr.low });
        }
    }

    return fractals;
}

// 生成笔 (连接顶底分型)
function generateStrokes(fractals, data) {
    if (fractals.length < 2) return [];

    const strokes = [];
    let lastFractal = fractals[0];

    for (let i = 1; i < fractals.length; i++) {
        const current = fractals[i];

        // 必须是顶底交替
        if (current.type !== lastFractal.type) {
            const direction = lastFractal.type === 'bottom' ? 'up' : 'down';
            const startIdx = lastFractal.index;
            const endIdx = current.index;

            // 至少5根K线
            if (endIdx - startIdx >= 4) {
                strokes.push({
                    type: direction,
                    startIdx,
                    endIdx,
                    startDate: lastFractal.date,
                    endDate: current.date,
                    startPrice: lastFractal.price,
                    endPrice: current.price
                });
            }

            lastFractal = current;
        }
    }

    return strokes;
}

// 识别中枢 (3笔重叠区域)
function findPivots(strokes, data) {
    if (strokes.length < 3) return [];

    const pivots = [];

    for (let i = 0; i < strokes.length - 2; i++) {
        const s1 = strokes[i];
        const s2 = strokes[i + 1];
        const s3 = strokes[i + 2];

        // 获取3笔的价格范围
        const prices = [];
        for (let j = s1.startIdx; j <= s3.endIdx && j < data.length; j++) {
            prices.push(data[j].high, data[j].low);
        }

        if (prices.length > 0) {
            const high = Math.max(...prices);
            const low = Math.min(...prices);
            const range = high - low;

            // 中枢成立条件: 3笔有重叠
            if (range > 0) {
                pivots.push({
                    startIdx: s1.startIdx,
                    endIdx: s3.endIdx,
                    high,
                    low,
                    middle: (high + low) / 2
                });
            }
        }
    }

    return pivots;
}

// 转换为周线数据
function resampleToWeekly(dailyData) {
    const weeklyData = [];
    let currentWeek = null;
    let lastWeekKey = null;

    for (const d of dailyData) {
        const date = new Date(d.date);
        const weekKey = `${date.getFullYear()}-W${getWeekNumber(date)}`;

        if (weekKey !== lastWeekKey) {
            if (currentWeek) {
                weeklyData.push(currentWeek);
            }
            currentWeek = {
                date: d.date,
                open: d.open,
                high: d.high,
                low: d.low,
                close: d.close,
                volume: d.volume
            };
        } else {
            currentWeek.high = Math.max(currentWeek.high, d.high);
            currentWeek.low = Math.min(currentWeek.low, d.low);
            currentWeek.close = d.close;
            currentWeek.volume += d.volume;
        }

        lastWeekKey = weekKey;
    }

    if (currentWeek) {
        weeklyData.push(currentWeek);
    }

    return weeklyData;
}

function getWeekNumber(d) {
    const oneJan = new Date(d.getFullYear(), 0, 1);
    const numberOfDays = Math.floor((d - oneJan) / (24 * 60 * 60 * 1000));
    return Math.ceil((d.getDay() + 1 + numberOfDays) / 7);
}

// 加载CSV数据
function loadCSV(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const lines = content.trim().split('\n');

    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        const obj = {
            date: values[0],
            open: parseFloat(values[1]),
            high: parseFloat(values[2]),
            low: parseFloat(values[3]),
            close: parseFloat(values[4]),
            amount: parseFloat(values[5]),
            volume: parseFloat(values[6])
        };
        data.push(obj);
    }

    return data;
}

// 检查MACD顶背离 (价格创新高但MACD降低)
function checkTopDivergence(data, macdData, index, lookback = 20) {
    if (index < lookback + 5) return false;

    const recent = data.slice(index - lookback, index + 1);
    const highIdx = recent.findIndex(d => d.high === Math.max(...recent.map(r => r.high)));
    const currentHighIdx = recent.length - 1;

    // 当前价格创新高
    if (currentHighIdx !== highIdx) return false;

    // 检查MACD是否降低
    const prevPeakIdx = index - lookback + highIdx;
    if (prevPeakIdx < 0 || index >= macdData.macd.length) return false;

    const currentMACD = macdData.macd[index];
    const prevMACD = macdData.macd[prevPeakIdx];

    return currentMACD < prevMACD * 0.95;
}

// 运行回测 - 缠论策略
function runChanLunBacktest() {
    console.log('='.repeat(60));
    console.log('002600 驰宏锌锗 缠论策略回测');
    console.log('='.repeat(60));
    console.log('策略: 周线2买买入, 跌破1买止损, 日线MACD顶背离减仓50%, 2卖清仓');
    console.log('='.repeat(60));

    const csvFile = path.join(__dirname, 'test_output', 'sz002600.day.csv');
    const dailyData = loadCSV(csvFile);

    console.log(`\n数据概览:`);
    console.log(`  日期范围: ${dailyData[0].date} ~ ${dailyData[dailyData.length - 1].date}`);
    console.log(`  K线数量: ${dailyData.length} 条`);

    const buyHoldReturn = ((dailyData[dailyData.length - 1].close / dailyData[0].close - 1) * 100).toFixed(2);
    console.log(`  买入持有收益: ${buyHoldReturn}%`);

    // 生成周线
    const weeklyData = resampleToWeekly(dailyData);
    console.log(`  周线数量: ${weeklyData.length} 条`);

    // 计算MACD
    const dailyMACD = calculateMACD(dailyData);
    const weeklyMACD = calculateMACD(weeklyData);

    // 识别分型
    const dailyFractals = findFractals(dailyData);
    const weeklyFractals = findFractals(weeklyData);

    console.log(`\n缠论结构识别:`);
    console.log(`  日线分型: ${dailyFractals.length} 个`);
    console.log(`  周线分型: ${weeklyFractals.length} 个`);

    // 生成笔
    const dailyStrokes = generateStrokes(dailyFractals, dailyData);
    const weeklyStrokes = generateStrokes(weeklyFractals, weeklyData);

    console.log(`  日线笔: ${dailyStrokes.length} 个`);
    console.log(`  周线笔: ${weeklyStrokes.length} 个`);

    // 识别中枢
    const dailyPivots = findPivots(dailyStrokes, dailyData);
    const weeklyPivots = findPivots(weeklyStrokes, weeklyData);

    console.log(`  日线中枢: ${dailyPivots.length} 个`);
    console.log(`  周线中枢: ${weeklyPivots.length} 个`);

    console.log('\n' + '='.repeat(60));
    console.log('回测交易记录');
    console.log('='.repeat(60));

    const trades = [];
    let position = 0; // 0=空, 0.5=半仓, 1=满仓
    let entryPrice = 0; // 平均成本
    let entryDate = '';
    let entryIndex = 0;
    let firstBuyPrice = 0; // 1买价格(止损参考)
    let lastBuyPrice = 0; // 最近买入价

    // 周线底分型列表 (用于2买判断)
    const weeklyBottoms = weeklyFractals.filter(f => f.type === 'bottom');

    for (let i = 100; i < dailyData.length; i++) {
        const curr = dailyData[i];
        const currDate = new Date(curr.date);
        const currWeekKey = `${currDate.getFullYear()}-W${getWeekNumber(currDate)}`;

        const currDif = dailyMACD.dif[i] || 0;
        const currDea = dailyMACD.dea[i] || 0;
        const prevDif = dailyMACD.dif[i - 1] || 0;
        const prevDea = dailyMACD.dea[i - 1] || 0;

        // === 买入信号 ===
        if (position === 0) {
            // 检查周线2买: 周线出现底分型后，日线回踩不破底分型低点
            for (const wb of weeklyBottoms) {
                const wbDate = new Date(wb.date);
                const wbWeekKey = `${wbDate.getFullYear()}-W${getWeekNumber(wbDate)}`;

                // 周线底分型在10周内
                if (parseInt(currWeekKey.split('-')[1]) - parseInt(wbWeekKey.split('-')[1]) <= 10) {
                    // 日线MACD金叉确认
                    if (currDif > currDea && prevDif <= prevDea) {
                        // 当前价格高于周线底分型价格
                        if (curr.close > wb.price * 1.02) {
                            position = 1;
                            entryPrice = curr.close;
                            entryDate = curr.date;
                            entryIndex = i;
                            firstBuyPrice = wb.price;
                            lastBuyPrice = curr.close;

                            console.log(`[满仓买入] ${curr.date} @ ${curr.close.toFixed(2)} (周线2买, 底分型${wb.date} ${wb.price.toFixed(2)})`);
                            break;
                        }
                    }
                }
            }
        }

        // === 卖出信号 ===
        if (position > 0) {
            let sellSignal = false;
            let sellReason = '';
            let sellRatio = 0;

            // 1. 止损: 跌破1买价格
            if (curr.close < firstBuyPrice) {
                sellSignal = true;
                sellReason = '止损(破1买)';
                sellRatio = 1;
            }
            // 2. 日线MACD顶背离 - 减仓50%
            else if (position === 1 && checkTopDivergence(dailyData, dailyMACD, i)) {
                position = 0.5;
                const partialProfit = ((curr.close - entryPrice) / entryPrice * 100);
                console.log(`[减仓50%] ${curr.date} @ ${curr.close.toFixed(2)} (MACD顶背离, 已实现${partialProfit.toFixed(2)}%)`);
                entryPrice = (entryPrice + curr.close) / 2; // 更新平均成本
                continue;
            }
            // 3. 日线2卖 (MACD死叉) - 清仓
            else if (currDif < currDea && prevDif >= prevDea) {
                sellSignal = true;
                sellReason = '2卖(MACD死叉)';
                sellRatio = position; // 卖出剩余部分
            }

            if (sellSignal) {
                const exitPrice = curr.close;
                const profit = ((exitPrice - lastBuyPrice) / lastBuyPrice * 100);
                const holdDays = i - entryIndex;

                trades.push({
                    entryDate,
                    entryPrice: lastBuyPrice,
                    exitDate: curr.date,
                    exitPrice,
                    profit,
                    reason: sellReason,
                    holdDays,
                    position
                });

                console.log(`[清仓] ${curr.date} @ ${exitPrice.toFixed(2)} (${sellReason}) 收益: ${profit.toFixed(2)}% (${holdDays}天)`);
                position = 0;
                entryPrice = 0;
                firstBuyPrice = 0;
                lastBuyPrice = 0;
            }
        }
    }

    // 统计结果
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

runChanLunBacktest();
