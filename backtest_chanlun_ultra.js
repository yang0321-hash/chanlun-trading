#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// ============================================================
// 超级优化版缠论策略配置
// ============================================================
const CONFIG = {
    // 市场状态识别
    marketRegime: {
        enabled: true,
        lookback: 60,              // 市场状态判断周期
        trendThreshold: 0.15,      // 趋势阈值(15%涨幅定义为趋势)
        volatilityThreshold: 0.03  // 波动率阈值
    },

    // 动态ATR止损
    atrStop: {
        baseMultiplier: 1.5,       // 基础倍数
        volatilityAdjust: true,    // 根据波动率调整
        tightMultiplier: 1.0,      // 低波动时使用(紧止损)
        wideMultiplier: 2.5        // 高波动时使用(宽止损)
    },

    // 自适应金字塔加仓
    pyramid: {
        enabled: true,
        maxPositions: 3,
        // 根据趋势强度动态调整
        strongTrend: { threshold: 0.5, addRatio: 1.0, pullback: 0.03 },
        normalTrend: { threshold: 0.2, addRatio: 0.5, pullback: 0.02 },
        weakTrend: { threshold: 0.0, addRatio: 0.3, pullback: 0.015 }
    },

    // 智能移动止损
    trailingStop: {
        enabled: true,
        modes: {
            conservative: { activation: 0.05, trail: 0.03 },  // 保守: 5%激活, 3%回撤
            moderate: { activation: 0.10, trail: 0.05 },       // 中等: 10%激活, 5%回撤
            aggressive: { activation: 0.15, trail: 0.08 }       // 激进: 15%激活, 8%回撤
        },
        autoSelect: true    // 根据市场状态自动选择模式
    },

    // 多周期确认
    multiTimeframe: {
        enabled: true,
        requireWeeklyTrend: true,    // 要求周线趋势向上
        weeklyLookback: 20           // 周线确认周期
    },

    // 入场过滤
    entryFilter: {
        minVolumeRatio: 0.8,         // 最小成交量比例(相对于20日均量)
        requireMACDHistogram: true,  // 要求MACD柱状图连续2日上涨
        checkMarketSentiment: true   // 检查市场整体情绪
    },

    // 出场策略
    exitStrategy: {
        priority: ['trailing', 'topFractal', 'macdDeath', 'atrStop'],
        topFractalProfitThreshold: 0.12,  // 顶分型止盈阈值
        partialProfitTaking: {            // 分批止盈
            enabled: true,
            levels: [
                { profit: 0.10, ratio: 0.3 },   // 盈利10%减仓30%
                { profit: 0.20, ratio: 0.3 },   // 盈利20%再减30%
                { profit: 0.30, ratio: 0.5 }    // 盈利30%清仓50%
            ]
        }
    },

    // 风险管理
    risk: {
        maxPositions: 3,            // 最大同时持仓数
        maxDailyLoss: 0.05,         // 单日最大亏损5%暂停
        correlationLimit: 0.7       // 持仓相关性限制
    }
};

// ============================================================
// 核心指标计算
// ============================================================

// ATR计算 (增强版 - 使用真实波动范围)
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
            atr.push(tr.slice(0, i + 1).reduce((a, b) => a + b, 0) / (i + 1));
        } else if (i === period - 1) {
            atr.push(tr.slice(0, period).reduce((a, b) => a + b, 0) / period);
        } else {
            atr.push((atr[i - 1] * (period - 1) + tr[i]) / period);
        }
    }

    return atr;
}

// 波动率计算 (用于自适应参数)
function calculateVolatility(data, period = 20) {
    const returns = [];
    for (let i = 1; i < data.length; i++) {
        returns.push(Math.log(data[i].close / data[i - 1].close));
    }

    const vol = [];
    for (let i = 0; i < returns.length; i++) {
        if (i < period - 1) {
            const slice = returns.slice(0, i + 1);
            const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
            const variance = slice.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / slice.length;
            vol.push(Math.sqrt(variance) * Math.sqrt(252)); // 年化波动率
        } else {
            const slice = returns.slice(i - period + 1, i + 1);
            const mean = slice.reduce((a, b) => a + b, 0) / period;
            const variance = slice.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / period;
            vol.push(Math.sqrt(variance) * Math.sqrt(252));
        }
    }

    // 补齐前面
    while (vol.length < data.length) {
        vol.unshift(vol[0] || 0.02);
    }

    return vol;
}

// MACD计算
function calculateMACD(data, fast = 12, slow = 26, signal = 9) {
    const closes = data.map(d => d.close);
    const emaFast = calculateEMA(closes, fast);
    const emaSlow = calculateEMA(closes, slow);
    const dif = emaFast.map((v, i) => v - emaSlow[i]);
    const dea = calculateEMA(dif, signal);
    const macd = dif.map((v, i) => (v - dea[i]) * 2);
    const histogram = macd.map((v, i) => v * 2); // 柱状图
    return { dif, dea, macd, histogram };
}

function calculateEMA(data, period) {
    const result = [];
    const mult = 2 / (period + 1);
    for (let i = 0; i < data.length; i++) {
        if (i < period) {
            const sum = data.slice(0, i + 1).reduce((a, b) => a + b, 0);
            result.push(sum / (i + 1));
        } else {
            result.push((data[i] - result[i - 1]) * mult + result[i - 1]);
        }
    }
    return result;
}

// 趋势强度计算 (ADX简化版)
function calculateTrendStrength(data, period = 14) {
    const strength = [];

    for (let i = 0; i < data.length; i++) {
        if (i < period + 1) {
            strength.push(0);
        } else {
            // 计算方向移动
            const upMoves = [], downMoves = [];
            for (let j = i - period; j < i; j++) {
                if (data[j].close > data[j - 1]?.close) {
                    upMoves.push(data[j].close - data[j - 1].close);
                } else if (data[j].close < data[j - 1]?.close) {
                    downMoves.push(data[j - 1].close - data[j].close);
                }
            }

            const avgUp = upMoves.length > 0 ? upMoves.reduce((a, b) => a + b, 0) / period : 0;
            const avgDown = downMoves.length > 0 ? downMoves.reduce((a, b) => a + b, 0) / period : 0;

            const dx = period === 0 ? 0 : Math.abs(avgUp - avgDown) / (avgUp + avgDown) * 100;
            strength.push(Math.min(100, Math.max(0, dx)));
        }
    }

    return strength;
}

// 分型识别
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

// ============================================================
// 市场状态识别
// ============================================================

function detectMarketRegime(data, volatility, idx) {
    const lookback = CONFIG.marketRegime.lookback;
    const start = Math.max(0, idx - lookback);
    const slice = data.slice(start, idx + 1);

    if (slice.length < lookback / 2) return 'unknown';

    const firstPrice = slice[0].close;
    const lastPrice = slice[slice.length - 1].close;
    const totalReturn = (lastPrice - firstPrice) / firstPrice;

    // 计算价格波动范围
    const highs = slice.map(d => d.high);
    const lows = slice.map(d => d.low);
    const priceRange = (Math.max(...highs) - Math.min(...lows)) / firstPrice;

    const avgVol = volatility.slice(start, idx + 1).reduce((a, b) => a + b, 0) / (idx - start + 1);

    // 判断市场状态
    if (totalReturn > CONFIG.marketRegime.trendThreshold) {
        return 'uptrend';  // 上升趋势
    } else if (totalReturn < -CONFIG.marketRegime.trendThreshold) {
        return 'downtrend'; // 下降趋势
    } else if (priceRange < CONFIG.marketRegime.volatilityThreshold * 2) {
        return 'ranging';   // 震荡市
    } else if (avgVol > 0.4) {
        return 'volatile';  // 高波动
    } else {
        return 'normal';    // 正常
    }
}

// ============================================================
// 交易信号
// ============================================================

// 检查2买信号
function checkSecondBuy(data, fractals, idx, lookback = 60) {
    for (let i = fractals.length - 1; i >= 0; i--) {
        const f = fractals[i];
        if (f.type === 'bottom' && f.index < idx && idx - f.index <= lookback) {
            const lowSinceBottom = Math.min(...data.slice(f.index, idx).map(d => d.low));
            if (data[idx].close > f.price && lowSinceBottom > f.price * 0.97) {
                return { bottomPrice: f.price, bottomIdx: f.index, strength: (data[idx].close - f.price) / f.price };
            }
        }
    }
    return null;
}

// 多周期确认
function checkMultiTimeframe(dailyData, weeklyData, dailyIdx) {
    if (!CONFIG.multiTimeframe.enabled) return true;

    // 计算对应的周线索引
    const weeklyIdx = Math.floor(dailyIdx / 5);
    if (weeklyIdx < CONFIG.multiTimeframe.weeklyLookback) return false;

    // 检查周线趋势
    const weeklySlice = weeklyData.slice(weeklyIdx - CONFIG.multiTimeframe.weeklyLookback, weeklyIdx + 1);
    const weeklyTrend = (weeklySlice[weeklySlice.length - 1].close - weeklySlice[0].close) / weeklySlice[0].close;

    if (CONFIG.multiTimeframe.requireWeeklyTrend && weeklyTrend < 0) {
        return false; // 周线下降，不做多
    }

    return true;
}

// 入场过滤
function checkEntryFilter(data, macdData, idx, atr) {
    const filter = CONFIG.entryFilter;

    // 成交量检查
    if (filter.minVolumeRatio > 0) {
        const recentVolumes = data.slice(Math.max(0, idx - 20), idx).map(d => d.volume);
        const avgVolume = recentVolumes.reduce((a, b) => a + b, 0) / recentVolumes.length;
        if (data[idx].volume < avgVolume * filter.minVolumeRatio) {
            return { pass: false, reason: '成交量不足' };
        }
    }

    // MACD柱状图确认
    if (filter.requireMACDHistogram && idx >= 2) {
        if (macdData.histogram[idx] <= macdData.histogram[idx - 1] ||
            macdData.histogram[idx - 1] <= macdData.histogram[idx - 2]) {
            return { pass: false, reason: 'MACD柱状图不强势' };
        }
    }

    return { pass: true };
}

// ============================================================
// 回测引擎
// ============================================================

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

// 生成周线数据
function generateWeeklyData(dailyData) {
    const weekly = [];
    let weekData = null;

    for (let i = 0; i < dailyData.length; i++) {
        if (i % 5 === 0) {
            if (weekData) weekly.push(weekData);
            weekData = {
                date: dailyData[i].date,
                open: dailyData[i].open,
                high: dailyData[i].high,
                low: dailyData[i].low,
                close: dailyData[i].close,
                volume: dailyData[i].volume
            };
        } else if (weekData) {
            weekData.high = Math.max(weekData.high, dailyData[i].high);
            weekData.low = Math.min(weekData.low, dailyData[i].low);
            weekData.close = dailyData[i].close;
            weekData.volume += dailyData[i].volume;
        }
    }
    if (weekData) weekly.push(weekData);
    return weekly;
}

// 超级优化策略回测
function backtestUltra(dailyData, weeklyData, atr, volatility, macdData, fractals, trendStrength) {
    const trades = [];
    let state = {
        inPosition: false,
        entries: [],
        totalShares: 0,
        avgPrice: 0,
        highestPrice: 0,
        highestProfit: 0,
        stopLoss: 0,
        trailingStop: 0,
        trailingMode: 'moderate',
        entryDate: '',
        entryIndex: 0,
        bottomPrice: 0,
        partialExit: { taken: [], remaining: 1.0 }
    };

    for (let i = 50; i < dailyData.length; i++) {
        const curr = dailyData[i];
        const currDif = macdData.dif[i] || 0;
        const currDea = macdData.dea[i] || 0;
        const prevDif = macdData.dif[i - 1] || 0;
        const prevDea = macdData.dea[i - 1] || 0;
        const goldenCross = currDif > currDea && prevDif <= prevDea;
        const deathCross = currDif < currDea && prevDif >= prevDea;

        // 市场状态
        const regime = detectMarketRegime(dailyData, volatility, i);
        const trendStr = trendStrength[i] || 0;

        // === 入场逻辑 ===
        if (!state.inPosition) {
            const secondBuy = checkSecondBuy(dailyData, fractals, i);
            if (!secondBuy) continue;

            // 多周期确认
            if (!checkMultiTimeframe(dailyData, weeklyData, i)) continue;

            // MACD金叉确认
            if (!goldenCross) continue;

            // 入场过滤
            const entryFilter = checkEntryFilter(dailyData, macdData, i, atr);
            if (!entryFilter.pass) continue;

            // 市场状态过滤
            if (regime === 'downtrend') continue;

            // 计算动态止损
            const currentAtr = atr[i];
            const currentVol = volatility[i];

            let atrMult = CONFIG.atrStop.baseMultiplier;
            if (CONFIG.atrStop.volatilityAdjust) {
                if (currentVol < 0.2) atrMult = CONFIG.atrStop.tightMultiplier;
                else if (currentVol > 0.4) atrMult = CONFIG.atrStop.wideMultiplier;
            }

            const atrStop = curr.close - currentAtr * atrMult;

            state = {
                inPosition: true,
                entries: [{ price: curr.close, shares: 1, date: curr.date, atr: currentAtr }],
                totalShares: 1,
                avgPrice: curr.close,
                highestPrice: curr.close,
                highestProfit: 0,
                stopLoss: Math.max(atrStop, secondBuy.bottomPrice * 0.97),
                trailingStop: 0,
                trailingMode: regime === 'uptrend' ? 'aggressive' : 'moderate',
                entryDate: curr.date,
                entryIndex: i,
                bottomPrice: secondBuy.bottomPrice,
                regime: regime,
                partialExit: { taken: [], remaining: 1.0 }
            };
        }

        // === 持仓管理 ===
        if (state.inPosition) {
            const currentProfit = (curr.close - state.avgPrice) / state.avgPrice;
            const remaining = state.partialExit.remaining;

            // 更新最高价
            if (curr.close > state.highestPrice) {
                state.highestPrice = curr.close;
                state.highestProfit = (state.highestPrice - state.avgPrice) / state.avgPrice;
            }

            // 分批止盈
            if (CONFIG.exitStrategy.partialProfitTaking.enabled && remaining > 0.1) {
                for (const level of CONFIG.exitStrategy.partialProfitTaking.levels) {
                    if (currentProfit >= level.profit && !state.partialExit.taken.includes(level.profit)) {
                        state.partialExit.taken.push(level.profit);
                        state.partialExit.remaining -= level.ratio;

                        // 部分止盈记录
                        const partialProfit = currentProfit * level.ratio * 100;
                        trades.push({
                            entryDate: state.entryDate,
                            exitDate: curr.date,
                            entryPrice: state.avgPrice,
                            exitPrice: curr.close,
                            profit: partialProfit,
                            holdDays: i - state.entryIndex,
                            reason: `部分止盈${(level.profit * 100).toFixed(0)}%`,
                            isPartial: true,
                            remaining: state.partialExit.remaining
                        });

                        // 调整止损到成本价
                        state.stopLoss = Math.max(state.stopLoss, state.avgPrice);
                    }
                }
            }

            // 金字塔加仓
            if (CONFIG.pyramid.enabled && state.entries.length < CONFIG.pyramid.maxPositions && remaining > 0.3) {
                if (currentProfit > 0.03) {
                    let pyrConfig;
                    if (trendStr > 50) pyrConfig = CONFIG.pyramid.strongTrend;
                    else if (trendStr > 25) pyrConfig = CONFIG.pyramid.normalTrend;
                    else pyrConfig = CONFIG.pyramid.weakTrend;

                    if (currentProfit >= pyrConfig.threshold) {
                        const pullback = (state.highestPrice - curr.close) / state.highestPrice;
                        if (pullback >= pyrConfig.pullback && pullback <= 0.05) {
                            const addShares = pyrConfig.addRatio;
                            const newAvg = (state.avgPrice * state.totalShares + curr.close * addShares) /
                                         (state.totalShares + addShares);

                            state.entries.push({ price: curr.close, shares: addShares, date: curr.date });
                            state.totalShares += addShares;
                            state.avgPrice = newAvg;

                            // 更新止损
                            const newAtrStop = curr.close - atr[i] * CONFIG.atrStop.baseMultiplier;
                            state.stopLoss = Math.max(state.stopLoss, newAtrStop);
                        }
                    }
                }
            }

            // 移动止损
            if (CONFIG.trailingStop.enabled) {
                let modeConfig;
                if (CONFIG.trailingStop.autoSelect) {
                    if (state.trailingMode === 'aggressive') modeConfig = CONFIG.trailingStop.modes.aggressive;
                    else if (state.trailingMode === 'conservative') modeConfig = CONFIG.trailingStop.modes.conservative;
                    else modeConfig = CONFIG.trailingStop.modes.moderate;
                } else {
                    modeConfig = CONFIG.trailingStop.modes.moderate;
                }

                if (currentProfit >= modeConfig.activation) {
                    const newTrailStop = state.highestPrice * (1 - modeConfig.trail);
                    if (newTrailStop > state.trailingStop) {
                        state.trailingStop = newTrailStop;
                    }
                }
            }

            // === 出场判断 ===
            let exitSignal = false;
            let exitReason = '';
            let exitPrice = curr.close;

            // 优先级1: 移动止损
            if (state.trailingStop > 0 && curr.close < state.trailingStop) {
                exitSignal = true;
                exitReason = '移动止损';
                exitPrice = state.trailingStop;
            }
            // 优先级2: 顶分型止盈
            else if (currentProfit > CONFIG.exitStrategy.topFractalProfitThreshold) {
                const recentTop = fractals.filter(f =>
                    f.type === 'top' && f.index >= i - 3 && f.index <= i
                );
                if (recentTop.length > 0) {
                    exitSignal = true;
                    exitReason = '顶分型';
                }
            }
            // 优先级3: MACD死叉
            else if (deathCross && currentProfit > 0.02) {
                exitSignal = true;
                exitReason = 'MACD死叉';
            }
            // 优先级4: ATR止损
            else if (curr.close < state.stopLoss) {
                exitSignal = true;
                exitReason = 'ATR止损';
            }

            if (exitSignal && remaining > 0) {
                const finalProfit = (exitPrice - state.entries[0].price) / state.entries[0].price * 100 * remaining;

                if (!exitReason.includes('部分')) {
                    trades.push({
                        entryDate: state.entryDate,
                        exitDate: curr.date,
                        entryPrice: state.entries[0].price,
                        exitPrice: exitPrice,
                        profit: finalProfit,
                        holdDays: i - state.entryIndex,
                        reason: exitReason,
                        addCount: state.entries.length - 1,
                        maxProfit: state.highestProfit * 100,
                        regime: state.regime
                    });

                    state = {
                        inPosition: false,
                        entries: [], totalShares: 0, avgPrice: 0,
                        highestPrice: 0, highestProfit: 0,
                        stopLoss: 0, trailingStop: 0,
                        trailingMode: 'moderate',
                        entryDate: '', entryIndex: 0, bottomPrice: 0,
                        partialExit: { taken: [], remaining: 1.0 }
                    };
                }
            }
        }
    }

    return trades;
}

// 原版策略
function backtestOriginal(dailyData, macdData, fractals) {
    const trades = [];
    let pos = false, entryPrice = 0, entryDate = '', entryIdx = 0, bottomPrice = 0;

    for (let i = 50; i < dailyData.length; i++) {
        const goldenCross = (macdData.dif[i] || 0) > (macdData.dea[i] || 0) &&
                           (macdData.dif[i - 1] || 0) <= (macdData.dea[i - 1] || 0);
        const deathCross = (macdData.dif[i] || 0) < (macdData.dea[i] || 0) &&
                           (macdData.dif[i - 1] || 0) >= (macdData.dea[i - 1] || 0);

        if (!pos) {
            const sb = checkSecondBuy(dailyData, fractals, i);
            if (sb && goldenCross) {
                pos = true; entryPrice = dailyData[i].close;
                entryDate = dailyData[i].date; entryIdx = i;
                bottomPrice = sb.bottomPrice;
            }
        } else {
            let sell = false, reason = '';
            if (dailyData[i].close < bottomPrice) { sell = true; reason = '止损'; }
            else if (deathCross) { sell = true; reason = 'MACD死叉'; }
            if (sell) {
                trades.push({
                    entryDate, exitDate: dailyData[i].date,
                    profit: (dailyData[i].close / entryPrice - 1) * 100,
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
    console.log('缠论策略超级优化版 - 原版 vs 增强版 vs 超级版');
    console.log('='.repeat(85));
    console.log('\n新增优化:');
    console.log('  1. 市场状态识别 (趋势/震荡/高波动 - 自适应调整参数)');
    console.log('  2. 动态ATR止损 (根据波动率自动调整止损倍数)');
    console.log('  3. 自适应金字塔 (根据趋势强度调整加仓比例)');
    console.log('  4. 智能移动止损 (根据市场状态选择保守/激进模式)');
    console.log('  5. 多周期确认 (周线趋势过滤)');
    console.log('  6. 入场过滤 (成交量/ MACD柱状图确认)');
    console.log('  7. 分批止盈 (10%/20%/30%分批减仓)');
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

        const dailyData = loadCSV(csvFile);
        const weeklyData = generateWeeklyData(dailyData);
        const atr = calculateATR(dailyData);
        const volatility = calculateVolatility(dailyData);
        const macdData = calculateMACD(dailyData);
        const fractals = findFractals(dailyData);
        const trendStrength = calculateTrendStrength(dailyData);

        const buyHold = ((dailyData[dailyData.length - 1].close / dailyData[0].close - 1) * 100);
        const origTrades = backtestOriginal(dailyData, macdData, fractals);
        const ultraTrades = backtestUltra(dailyData, weeklyData, atr, volatility, macdData, fractals, trendStrength);

        const origTotal = origTrades.reduce((s, t) => s + t.profit, 0);
        const ultraTotal = ultraTrades.filter(t => !t.isPartial).reduce((s, t) => s + t.profit, 0);
        const ultraTotalWithPartial = ultraTrades.reduce((s, t) => s + t.profit, 0);

        results.push({
            code: stock.code,
            name: stock.name,
            buyHold,
            orig: origTotal,
            ultra: ultraTotal,
            ultraWithPartial: ultraTotalWithPartial,
            trades: ultraTrades.filter(t => !t.isPartial).length,
            origTrades: origTrades.length
        });
    }

    console.log('\n┌────────────┬──────────┬─────────┬─────────┬─────────┬─────────┐');
    console.log('│ 股票       │ 买入持有 │ 原版    │ 超级版  │ 超级+   │ 交易数  │');
    console.log('│            │ 收益%    │ 收益%   │ 收益%   │ 分批%   │ (超级)  │');
    console.log('├────────────┼─────────┼─────────┼─────────┼─────────┼─────────┤');

    for (const r of results) {
        const improve = ((r.ultra - r.orig) / (Math.abs(r.orig) || 1) * 100).toFixed(0);
        const mark = r.ultra > r.orig ? '+' : '';
        console.log(`│ ${(r.code).padEnd(10)} │ ${r.buyHold.toFixed(5).padStart(7)} │ ${r.orig.toFixed(2).padStart(7)} │ ${r.ultra.toFixed(2).padStart(7)} │ ${r.ultraWithPartial.toFixed(2).padStart(7)} │ ${r.trades.toString().padStart(7)} │`);
    }

    console.log('└────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘');

    // 统计
    const avgOrig = results.reduce((s, r) => s + r.orig, 0) / results.length;
    const avgUltra = results.reduce((s, r) => s + r.ultra, 0) / results.length;
    const beatCount = results.filter(r => r.ultra > r.orig).length;

    console.log(`\n平均收益:`);
    console.log(`  原版:     ${avgOrig.toFixed(2)}%`);
    console.log(`  超级版:   ${avgUltra.toFixed(2)}% (改善 ${(avgUltra - avgOrig).toFixed(2)}%)`);
    console.log(`  超越原版: ${beatCount}/${results.length} 只股票`);

    // 胜率分析
    console.log('\n' + '='.repeat(85));
    console.log('详细分析 - 超级版策略交易特征');
    console.log('='.repeat(85));

    // 分析其中一只股票的详细交易
    const bestStock = results.reduce((a, b) => a.ultra > b.ultra ? a : b);
    console.log(`\n最佳表现: ${bestStock.code} ${bestStock.name}`);
    console.log(`原版收益: ${bestStock.orig.toFixed(2)}% -> 超级版: ${bestStock.ultra.toFixed(2)}%`);

    console.log('\n' + '='.repeat(85));
}

runComparison();
