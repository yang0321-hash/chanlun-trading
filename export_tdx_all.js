const fs = require('fs');
const path = require('path');

// 通达信数据路径
const TDX_VIPDOC_PATH = 'D:/new_tdx/vipdoc';

// 输出目录
const OUTPUT_DIR = 'D:/新建文件夹/claude/tdx_output';

// 样本输出目录
const SAMPLE_DIR = 'D:/新建文件夹/claude/tdx-parser-workspace/iteration-1/eval-3/without_skill/outputs';

// CSV表头
const CSV_HEADER = 'date,open,high,low,close,volume,amount';

function readInt32LE(buffer, offset) {
    return buffer.readInt32LE(offset);
}

function readFloat(buffer, offset) {
    return buffer.readFloatLE(offset);
}

function formatDate(dateVal) {
    const dateStr = dateVal.toString();
    if (dateStr.length === 8) {
        return `${dateStr.slice(0,4)}-${dateStr.slice(4,6)}-${dateStr.slice(6,8)}`;
    }
    return dateStr;
}

function parseDayFile(filepath) {
    const buffer = fs.readFileSync(filepath);
    const records = [];
    const recordSize = 32;

    for (let i = 0; i + recordSize <= buffer.length; i += recordSize) {
        const date = readInt32LE(buffer, i);
        const open = readInt32LE(buffer, i + 4) / 100;
        const high = readInt32LE(buffer, i + 8) / 100;
        const low = readInt32LE(buffer, i + 12) / 100;
        const close = readInt32LE(buffer, i + 16) / 100;
        const amount = readFloat(buffer, i + 20);
        const volume = readInt32LE(buffer, i + 24);
        // const reserved = readInt32LE(buffer, i + 28);

        records.push({
            date: formatDate(date),
            open: open.toFixed(2),
            high: high.toFixed(2),
            low: low.toFixed(2),
            close: close.toFixed(2),
            amount: amount.toFixed(2),
            volume: volume
        });
    }

    return records;
}

function toCSV(data, filepath) {
    if (data.length === 0) return false;

    try {
        const rows = data.map(r =>
            `${r.date},${r.open},${r.high},${r.low},${r.close},${r.volume},${r.amount}`
        );
        const csv = [CSV_HEADER, ...rows].join('\n');

        fs.mkdirSync(path.dirname(filepath), { recursive: true });
        fs.writeFileSync(filepath, csv, 'utf8');
        return true;
    } catch (e) {
        console.error(`  保存失败: ${e.message}`);
        return false;
    }
}

function exportMarket(market, outputSubdir, sampleDir) {
    const ldayPath = path.join(TDX_VIPDOC_PATH, market, 'lday');

    if (!fs.existsSync(ldayPath)) {
        console.log(`路径不存在: ${ldayPath}`);
        return 0;
    }

    const files = fs.readdirSync(ldayPath).filter(f => f.endsWith('.day'));
    const total = files.length;
    console.log(`\n${market.toUpperCase()} 市场: 发现 ${total} 个日线文件`);

    let successCount = 0;
    let errorCount = 0;
    let sampleCount = 0;

    for (let i = 0; i < total; i++) {
        const file = files[i];
        const inputPath = path.join(ldayPath, file);
        const stockCode = file.replace('.day', '');

        try {
            const data = parseDayFile(inputPath);

            if (data.length > 0) {
                // 输出到主目录
                const outputFile = path.join(OUTPUT_DIR, outputSubdir, `${stockCode}.csv`);
                if (toCSV(data, outputFile)) {
                    successCount++;

                    // 保存前5个文件到样本目录
                    if (sampleDir && sampleCount < 5) {
                        const sampleFile = path.join(sampleDir, `${market}_${stockCode}.csv`);
                        toCSV(data, sampleFile);
                        sampleCount++;
                    }
                }
            } else {
                errorCount++;
            }
        } catch (e) {
            console.error(`  解析失败: ${file} - ${e.message}`);
            errorCount++;
        }

        // 进度显示
        if ((i + 1) % 500 === 0 || (i + 1) === total) {
            console.log(`  进度: ${i + 1}/${total} (${Math.floor((i + 1) * 100 / total)}%)`);
        }
    }

    console.log(`  完成: 成功 ${successCount}, 失败 ${errorCount}`);
    return successCount;
}

function main() {
    console.log('='.repeat(60));
    console.log('       通达信日线数据导出工具');
    console.log('='.repeat(60));
    console.log(`源路径: ${TDX_VIPDOC_PATH}`);
    console.log(`输出目录: ${OUTPUT_DIR}`);
    console.log(`样本目录: ${SAMPLE_DIR}`);

    const startTime = Date.now();

    // 确保输出目录存在
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    fs.mkdirSync(SAMPLE_DIR, { recursive: true });

    // 导出上海市场
    const shCount = exportMarket('sh', 'sh', SAMPLE_DIR);

    // 导出深圳市场
    const szCount = exportMarket('sz', 'sz', SAMPLE_DIR);

    // 尝试导出其他市场（如果存在）
    const bjCount = exportMarket('bj', 'bj', null);
    const dsCount = exportMarket('ds', 'ds', null);

    const endTime = Date.now();
    const duration = ((endTime - startTime) / 1000).toFixed(1);

    console.log('\n' + '='.repeat(60));
    console.log('导出完成!');
    console.log(`上海市场: ${shCount} 个文件`);
    console.log(`深圳市场: ${szCount} 个文件`);
    console.log(`北京市场: ${bjCount} 个文件`);
    console.log(`大宗交易: ${dsCount} 个文件`);
    console.log(`总计: ${shCount + szCount + bjCount + dsCount} 个文件`);
    console.log(`耗时: ${duration} 秒`);
    console.log(`输出目录: ${OUTPUT_DIR}`);
    console.log(`样本文件: ${SAMPLE_DIR}`);
    console.log('='.repeat(60));
}

main();
