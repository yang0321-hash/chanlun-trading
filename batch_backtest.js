#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// 选定回测股票列表
const STOCKS = [
    { code: 'sh600519', name: '贵州茅台', sector: '消费' },
    { code: 'sz000858', name: '五粮液', sector: '消费' },
    { code: 'sh600036', name: '招商银行', sector: '金融' },
    { code: 'sz002594', name: '比亚迪', sector: '新能源' },
    { code: 'sz300750', name: '宁德时代', sector: '新能源' }
];

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
        const prev = data[i - 1], curr = data[i], next = data[i + 1];
        if (curr.high > prev.high && curr.high > next.high &&
            curr.low > prev.low && curr.low > next.low) {
            fractals.push({ index: i, type: 'top', price: curr.high });
        } else if (curr.low < prev.low && curr.low < next.low &&
                   curr.high < prev.high && curr.high < next.high) {
            fractals.push({ index: i, type: 'bottom', price: curr.low });
        }
    }
    return fractals;
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

// 检查2买信号
function checkSecondBuy(dailyData, dailyFractals, dailyIndex, lookback = 60) {
    let recentBottom = null;
    for (let i = dailyFractals.length - 1; i >= 0; i--) {
        const f = dailyFractals[i];
        if (f.type === 'bottom' && f.index < dailyIndex && dailyIndex - f.index <= lookback) {
            recentBottom = f;
            break;
        }
    }
    if (!recentBottom) return null;
    const current = dailyData[dailyIndex];
    const lowSinceBottom = Math.min(...dailyData.slice(recentBottom.index, dailyIndex).map(d => d.low));
    if (current.close > recentBottom.price && lowSinceBottom > recentBottom.price * 0.98) {
        return { bottomPrice: recentBottom.price, currentPrice: current.close };
    }
    return null;
}

// 策略1: MACD金死叉
function backtestMACD(data, macdData) {
    const trades = [];
    let position = false, entryPrice = 0, entryDate = '', entryIdx = 0;

    for (let i = 50; i < data.length; i++) {
        const currDif = macdData.dif[i] || 0, currDea = macdData.dea[i] || 0;
        const prevDif = macdData.dif[i - 1] || 0, prevDea = macdData.dea[i - 1] || 0;

        if (!position && currDif > currDea && prevDif <= prevDea) {
            position = true; entryPrice = data[i].close; entryDate = data[i].date; entryIdx = i;
        } else if (position) {
            let sell = false, reason = '';
            if (data[i].close < entryPrice * 0.95) { sell = true; reason = '止损'; }
            else if (currDif < currDea && prevDif >= prevDea) { sell = true; reason = 'MACD死叉'; }
            if (sell) {
                trades.push({ entryDate, exitDate: data[i].date, profit: (data[i].close / entryPrice - 1) * 100, holdDays: i - entryIdx });
                position = false;
            }
        }
    }
    return trades;
}

// 策略2: 缠论2买
function backtestChanLun(data, macdData, fractals) {
    const trades = [];
    let position = 0, entryPrice = 0, entryDate = '', entryIdx = 0, bottomPrice = 0;

    for (let i = 50; i < data.length; i++) {
        const currDif = macdData.dif[i] || 0, currDea = macdData.dea[i] || 0;
        const prevDif = macdData.dif[i - 1] || 0, prevDea = macdData.dea[i - 1] || 0;
        const goldenCross = currDif > currDea && prevDif <= prevDea;

        if (position === 0) {
            const secondBuy = checkSecondBuy(data, fractals, i);
            if (secondBuy && goldenCross) {
                position = 1; entryPrice = data[i].close; entryDate = data[i].date;
                entryIdx = i; bottomPrice = secondBuy.bottomPrice;
            }
        } else if (position > 0) {
            let sell = false, reason = '';
            if (data[i].close < bottomPrice) { sell = true; reason = '止损'; }
            else if (currDif < currDea && prevDif >= prevDea) { sell = true; reason = 'MACD死叉'; }
            if (sell) {
                trades.push({ entryDate, exitDate: data[i].date, profit: (data[i].close / entryPrice - 1) * 100, holdDays: i - entryIdx });
                position = 0; bottomPrice = 0;
            }
        }
    }
    return trades;
}

function runBatchBacktest() {
    console.log('='.repeat(80));
    console.log('A股多股票策略回测 - 普适性验证');
    console.log('='.repeat(80));

    const results = [];

    for (const stock of STOCKS) {
        const csvFile = path.join(__dirname, 'test_output', `${stock.code}.day.csv`);
        if (!fs.existsSync(csvFile)) {
            console.log(`\n[跳过] ${stock.code} ${stock.name} - 文件不存在`);
            continue;
        }

        const data = loadCSV(csvFile);
        const macdData = calculateMACD(data);
        const fractals = findFractals(data);

        const buyHold = ((data[data.length - 1].close / data[0].close - 1) * 100);
        const macdTrades = backtestMACD(data, macdData);
        const clTrades = backtestChanLun(data, macdData, fractals);

        const macdTotal = macdTrades.reduce((s, t) => s + t.profit, 0);
        const clTotal = clTrades.reduce((s, t) => s + t.profit, 0);

        results.push({
            code: stock.code,
            name: stock.name,
            sector: stock.sector,
            buyHold,
            macd: { trades: macdTrades.length, total: macdTotal, winRate: macdTrades.filter(t => t.profit > 0).length / macdTrades.length * 100 || 0 },
            chanlun: { trades: clTrades.length, total: clTotal, winRate: clTrades.filter(t => t.profit > 0).length / clTrades.length * 100 || 0 }
        });
    }

    console.log('\n' + '='.repeat(80));
    console.log('回测结果汇总');
    console.log('='.repeat(80));

    console.log('\n┌──────────────┬──────────┬─────────┬─────────┬─────────┬─────────┬─────────┐');
    console.log('│ 股票         │ 板块     │ 买入持有 │ MACD    │ MACD    │ 缠论    │ 缠论    │');
    console.log('│              │          │ 收益%   │ 收益%   │ 胜率%   │ 收益%   │ 胜率%   │');
    console.log('├──────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤');

    for (const r of results) {
        console.log(`│ ${r.code.padEnd(12)} │ ${(r.sector || '未知').padEnd(7)} │ ${r.buyHold.toFixed(2).padStart(7)} │ ${r.macd.total.toFixed(2).padStart(7)} │ ${r.macd.winRate.toFixed(0).padStart(7)} │ ${r.chanlun.total.toFixed(2).padStart(7)} │ ${r.chanlun.winRate.toFixed(0).padStart(7)} │`);
    }

    console.log('└──────────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘');

    // 统计
    const avgBuyHold = results.reduce((s, r) => s + r.buyHold, 0) / results.length;
    const avgMacd = results.reduce((s, r) => s + r.macd.total, 0) / results.length;
    const avgCl = results.reduce((s, r) => s + r.chanlun.total, 0) / results.length;

    console.log('\n平均收益:');
    console.log(`  买入持有: ${avgBuyHold.toFixed(2)}%`);
    console.log(`  MACD策略: ${avgMacd.toFixed(2)}% (${avgMacd > avgBuyHold ? '+' : ''}${(avgMacd - avgBuyHold).toFixed(2)}%)`);
    console.log(`  缠论策略: ${avgCl.toFixed(2)}% (${avgCl > avgBuyHold ? '+' : ''}${(avgCl - avgBuyHold).toFixed(2)}%)`);

    // 胜率统计
    const macdWins = results.filter(r => r.macd.total > r.buyHold).length;
    const clWins = results.filter(r => r.chanlun.total > r.buyHold).length;

    console.log(`\n超越买入持有比例:`);
    console.log(`  MACD策略: ${macdWins}/${results.length} (${(macdWins/results.length*100).toFixed(0)}%)`);
    console.log(`  缠论策略: ${clWins}/${results.length} (${(clWins/results.length*100).toFixed(0)}%)`);

    console.log('\n' + '='.repeat(80));
}

runBatchBacktest();
