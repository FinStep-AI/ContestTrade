
"""
Tushare数据提供模块
"""
import pandas as pd
from utils.tushare_utils import tushare_cached
from utils.tushare_utils import get_trade_date
from loguru import logger

class TushareDataProvider:
    @staticmethod
    def _get_tushare_data(func_name: str, func_kwargs: dict, trade_date: str) -> pd.DataFrame:
        """
        通用的Tushare数据获取函数

        Args:
            func_name: Tushare函数名
            func_kwargs: Tushare函数参数
            trade_date: 交易日期

        Returns:
            DataFrame: 获取到的数据
        """
        try:
            logger.info(f"获取 {trade_date} 的 {func_name} 数据")
            
            df = tushare_cached.run(
                func_name=func_name,
                func_kwargs=func_kwargs
            )
            
            if df is not None and not df.empty:
                if 'trade_date' not in df.columns:
                    df['trade_date'] = trade_date
                    
                logger.info(f"成功获取 {trade_date} {func_name} 数据，共 {len(df)} 条记录")
                return df
            else:
                logger.warning(f"{trade_date} 无 {func_name} 数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取 {trade_date} {func_name} 数据失败: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_hm_detail_data(trade_date: str) -> pd.DataFrame:
        """
        获取指定交易日的热门股明细数据
        """
        return TushareDataProvider._get_tushare_data(
            func_name="hm_detail",
            func_kwargs={"trade_date": trade_date},
            trade_date=trade_date
        )

    @staticmethod
    def get_kline_data(trade_date: str, kline_num: int = 90) -> dict:
        """
        获取三大指数的K线数据
        """
        try:
            logger.info(f"获取 {trade_date} 的三大指数K线数据")
            
            indices = [
                {"code": "000001.SH", "name": "上证指数"},
                {"code": "399006.SZ", "name": "创业板指"},
                {"code": "000688.SH", "name": "科创50"}
            ]
            
            end_datetime = pd.to_datetime(trade_date, format='%Y%m%d')
            start_datetime = end_datetime - pd.Timedelta(days=kline_num * 2)
            start_date = start_datetime.strftime('%Y%m%d')
            
            kline_data = {}
            
            for index in indices:
                stock_code = index["code"]
                stock_name = index["name"]
                
                try:
                    func_kwargs = {
                        'ts_code': stock_code,
                        'start_date': start_date,
                        'end_date': trade_date
                    }
                    df = tushare_cached.run("index_daily", func_kwargs=func_kwargs, verbose=False)
                    
                    if df is not None and not df.empty:
                        df = df.sort_values('trade_date', ascending=False)
                        df = df.head(kline_num)
                        df = df.sort_values('trade_date', ascending=True)
                        
                        kline_list = []
                        for _, row in df.iterrows():
                            kline_item = {
                                'trade_date': int(row['trade_date']),
                                'open_price': float(row['open']),
                                'high_price': float(row['high']),
                                'low_price': float(row['low']),
                                'close_price': float(row['close']),
                                'preclose_price': float(row['pre_close']),
                                'price_change': float(row['change']),
                                'price_change_rate': float(row['pct_chg']) / 100,
                                'trade_amount': float(row['amount']),
                                'trade_lots': float(row['vol'])
                            }
                            kline_list.append(kline_item)
                        
                        kline_data[stock_code] = {
                            'name': stock_name,
                            'data': kline_list
                        }
                        
                        logger.info(f"成功获取 {stock_name} K线数据，共 {len(kline_list)} 条记录")
                    else:
                        logger.warning(f"{stock_name} K线数据为空")
                        
                except Exception as e:
                    logger.error(f"获取 {stock_name} K线数据失败: {e}")
                    continue
            
            if kline_data:
                logger.info(f"成功获取三大指数K线数据，共 {len(kline_data)} 个指数")
                return kline_data
            else:
                logger.warning(f"{trade_date} 无K线数据")
                return {}
                
        except Exception as e:
            logger.error(f"获取K线数据失败: {e}")
            return {}

    @staticmethod
    def get_current_day_kline_data(trade_date: str) -> dict:
        """
        获取指定交易日的三大指数当日K线数据
        """
        try:
            logger.info(f"获取 {trade_date} 的三大指数当日数据")
            
            indices = [
                {"code": "000001.SH", "name": "上证指数"},
                {"code": "399006.SZ", "name": "创业板指"},
                {"code": "000688.SH", "name": "科创50"}
            ]
            
            current_day_data = {}
            
            for index in indices:
                stock_code = index["code"]
                stock_name = index["name"]
                
                try:
                    func_kwargs = {
                        'ts_code': stock_code,
                        'trade_date': trade_date
                    }
                    df = tushare_cached.run("index_daily", func_kwargs=func_kwargs, verbose=False)
                    
                    if df is not None and not df.empty:
                        row = df.iloc[0]
                        current_day_data[stock_code] = {
                            'name': stock_name,
                            'trade_date': int(row['trade_date']),
                            'open_price': float(row['open']),
                            'high_price': float(row['high']),
                            'low_price': float(row['low']),
                            'close_price': float(row['close']),
                            'preclose_price': float(row['pre_close']),
                            'price_change': float(row['change']),
                            'price_change_rate': float(row['pct_chg']) / 100,
                            'trade_amount': float(row['amount']),
                            'trade_lots': float(row['vol'])
                        }
                        
                        logger.info(f"成功获取 {stock_name} 当日数据")
                    else:
                        logger.warning(f"{stock_name} 当日数据为空")
                        
                except Exception as e:
                    logger.error(f"获取 {stock_name} 当日数据失败: {e}")
                    continue
            
            if current_day_data:
                logger.info(f"成功获取三大指数当日数据，共 {len(current_day_data)} 个指数")
                return current_day_data
            else:
                logger.warning(f"{trade_date} 无当日数据")
                return {}
                
        except Exception as e:
            logger.error(f"获取当日数据失败: {e}")
            return {}

    @staticmethod
    def get_limit_cpt_list_data(trade_date: str) -> pd.DataFrame:
        """
        获取指定交易日的最强板块统计数据
        """
        return TushareDataProvider._get_tushare_data(
            func_name="limit_cpt_list",
            func_kwargs={"trade_date": trade_date},
            trade_date=trade_date
        )

    @staticmethod
    def get_limit_list_d_data(trade_date: str, limit_type: str = 'U') -> pd.DataFrame:
        """
        获取指定交易日的涨跌停列表数据
        """
        return TushareDataProvider._get_tushare_data(
            func_name="limit_list_d",
            func_kwargs={
                "trade_date": trade_date,
                "limit_type": limit_type,
                "fields": "ts_code,trade_date,industry,name,close,pct_chg,open_times,up_stat,limit_times"
            },
            trade_date=trade_date
        )

    @staticmethod
    def get_limit_step_data(trade_date: str) -> pd.DataFrame:
        """
        获取指定交易日的连板天梯数据
        """
        return TushareDataProvider._get_tushare_data(
            func_name="limit_step",
            func_kwargs={"trade_date": trade_date},
            trade_date=trade_date
        )

    @staticmethod
    def get_sector_moneyflow_data(trade_date: str, top_n: int = 10) -> list:
        """
        获取指定交易日的板块资金流向数据（东方财富数据）
        """
        try:
            logger.info(f"获取 {trade_date} 的板块资金流向数据")
            
            func_kwargs = {
                'trade_date': trade_date,
                'fields': 'trade_date,ts_code,name,pct_change,close,net_amount,net_amount_rate,rank,buy_elg_amount,buy_lg_amount,buy_md_amount,buy_sm_amount,buy_sm_amount_stock'
            }
            df = tushare_cached.run("moneyflow_ind_dc", func_kwargs=func_kwargs, verbose=False)
            
            if df is not None and not df.empty:
                df = df.sort_values('net_amount', ascending=False).head(top_n)
                
                sectors_info = []
                for _, row in df.iterrows():
                    sector_info = {
                        'name': row['name'],
                        'ts_code': row['ts_code'],
                        'representative_stock': row.get('buy_sm_amount_stock', ''),
                        'close_price': float(row['close']),
                        'pct_change': float(row['pct_change']),
                        'net_amount': float(row['net_amount']),
                        'net_amount_rate': float(row['net_amount_rate']),
                        'buy_elg_amount': float(row['buy_elg_amount']),
                        'buy_lg_amount': float(row['buy_lg_amount']),
                        'buy_md_amount': float(row['buy_md_amount']),
                        'buy_sm_amount': float(row['buy_sm_amount']),
                        'rank': int(row['rank'])
                    }
                    sectors_info.append(sector_info)
                
                logger.info(f"成功获取板块资金流向数据，共 {len(sectors_info)} 个板块")
                return sectors_info
            else:
                logger.warning(f"{trade_date} 无板块资金流向数据")
                return []
                
        except Exception as e:
            logger.error(f"获取板块资金流向数据失败: {e}")
            return []

    @staticmethod
    def get_sector_moneyflow_summary(trade_date: str, top_n: int = 10) -> str:
        """
        获取指定交易日的板块资金流向数据摘要
        """
        try:
            sectors_info = TushareDataProvider.get_sector_moneyflow_data(trade_date, top_n)
            
            if not sectors_info:
                return None
            
            descriptions = []
            
            for i, sector in enumerate(sectors_info, 1):
                desc = f"{i}. {sector['name']}({sector['ts_code']})，排名第{sector['rank']}位，"
                desc += f"收盘{sector['close_price']:.2f}点，涨幅{sector['pct_change']:+.2f}%。"
                
                net_amount_billion = sector['net_amount'] / 100000000
                desc += f"净流入资金{net_amount_billion:+.1f}亿元(净流入比率{sector['net_amount_rate']:+.2f}%)，"
                
                elg_billion = sector['buy_elg_amount'] / 100000000
                lg_billion = sector['buy_lg_amount'] / 100000000
                md_billion = sector['buy_md_amount'] / 100000000
                sm_billion = sector['buy_sm_amount'] / 100000000
                
                desc += f"资金构成：超大单{elg_billion:.1f}亿、大单{lg_billion:.1f}亿、中单{md_billion:.1f}亿、小单{sm_billion:.1f}亿。"
                
                if sector['representative_stock']:
                    desc += f"代表股票：{sector['representative_stock']}。"
                
                descriptions.append(desc)
            
            if descriptions:
                return f"{trade_date}板块资金流向（净流入前{len(descriptions)}）：{' '.join(descriptions)}"
            else:
                return None
                
        except Exception as e:
            logger.error(f"获取板块资金流向数据摘要失败: {e}")
            return None

    @staticmethod
    def get_top_inst_data(trade_date: str) -> pd.DataFrame:
        """
        获取指定交易日的龙虎榜机构成交明细数据
        """
        return TushareDataProvider._get_tushare_data(
            func_name="top_inst",
            func_kwargs={"trade_date": trade_date},
            trade_date=trade_date
        )

    @staticmethod
    def get_top_list_data(trade_date: str) -> pd.DataFrame:
        """
        获取指定交易日的龙虎榜每日交易明细数据
        """
        return TushareDataProvider._get_tushare_data(
            func_name="top_list",
            func_kwargs={"trade_date": trade_date},
            trade_date=trade_date
        )

    @staticmethod
    def get_data_by_date_range(data_fetcher, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
        """
        获取指定日期范围内的Tushare数据

        Args:
            data_fetcher: 数据获取函数 (例如, get_hm_detail_data)
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame: 合并后的数据
        """
        try:
            trade_dates = get_trade_date()
            target_dates = [d for d in trade_dates if start_date <= d <= end_date]
            
            logger.info(f"获取 {start_date} 到 {end_date} 的数据，共 {len(target_dates)} 个交易日")
            
            all_data = []
            for trade_date in target_dates:
                df = data_fetcher(trade_date, **kwargs)
                if not df.empty:
                    all_data.append(df)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                logger.info(f"成功获取数据，总计 {len(combined_df)} 条记录")
                return combined_df
            else:
                logger.warning("未获取到任何数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            return pd.DataFrame()
