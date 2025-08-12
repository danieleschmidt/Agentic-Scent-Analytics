"""
Internationalization and localization support for global deployment.
"""

import json
import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class SupportedLocale(Enum):
    """Supported locales for the platform."""
    ENGLISH_US = "en_US"
    SPANISH_ES = "es_ES"
    FRENCH_FR = "fr_FR"
    GERMAN_DE = "de_DE"
    JAPANESE_JP = "ja_JP"
    CHINESE_CN = "zh_CN"
    CHINESE_TW = "zh_TW"
    KOREAN_KR = "ko_KR"
    ITALIAN_IT = "it_IT"
    PORTUGUESE_BR = "pt_BR"
    RUSSIAN_RU = "ru_RU"
    HINDI_IN = "hi_IN"


@dataclass
class LocalizationConfig:
    """Configuration for localization."""
    default_locale: SupportedLocale = SupportedLocale.ENGLISH_US
    fallback_locale: SupportedLocale = SupportedLocale.ENGLISH_US
    translations_path: Path = Path(__file__).parent.parent / "translations"
    auto_detect_locale: bool = True
    date_format: Optional[str] = None
    time_format: Optional[str] = None
    number_format: Optional[str] = None


class I18nManager:
    """
    Internationalization and localization manager.
    """
    
    def __init__(self, config: Optional[LocalizationConfig] = None):
        self.config = config or LocalizationConfig()
        self.current_locale = self.config.default_locale
        self.translations: Dict[str, Dict[str, str]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Load all translations
        self._load_translations()
        
        # Auto-detect locale if enabled
        if self.config.auto_detect_locale:
            self._auto_detect_locale()
    
    def _load_translations(self):
        """Load translation files for all supported locales."""
        if not self.config.translations_path.exists():
            self._create_default_translations()
        
        for locale in SupportedLocale:
            translation_file = self.config.translations_path / f"{locale.value}.json"
            
            if translation_file.exists():
                try:
                    with open(translation_file, 'r', encoding='utf-8') as f:
                        self.translations[locale.value] = json.load(f)
                    self.logger.debug(f"Loaded translations for {locale.value}")
                except Exception as e:
                    self.logger.error(f"Failed to load translations for {locale.value}: {e}")
                    self.translations[locale.value] = {}
            else:
                self.translations[locale.value] = {}
        
        self.logger.info(f"Loaded translations for {len(self.translations)} locales")
    
    def _create_default_translations(self):
        """Create default translation files."""
        self.config.translations_path.mkdir(parents=True, exist_ok=True)
        
        # Base translations in English
        base_translations = {
            # System messages
            "system.startup": "System starting up",
            "system.shutdown": "System shutting down",
            "system.ready": "System ready",
            "system.error": "System error occurred",
            
            # Quality control messages
            "quality.batch_approved": "Batch approved for release",
            "quality.batch_rejected": "Batch rejected - quality issues detected",
            "quality.anomaly_detected": "Quality anomaly detected",
            "quality.contamination_found": "Contamination found in sample",
            "quality.analysis_complete": "Quality analysis complete",
            
            # Sensor messages
            "sensor.connected": "Sensor connected successfully",
            "sensor.disconnected": "Sensor disconnected",
            "sensor.calibration_needed": "Sensor calibration required",
            "sensor.reading_invalid": "Invalid sensor reading",
            "sensor.maintenance_due": "Sensor maintenance due",
            
            # Agent messages
            "agent.started": "Agent started successfully",
            "agent.stopped": "Agent stopped",
            "agent.analysis_started": "Starting analysis",
            "agent.analysis_completed": "Analysis completed",
            "agent.recommendation": "Recommendation",
            
            # Alerts and warnings
            "alert.critical": "Critical Alert",
            "alert.warning": "Warning",
            "alert.info": "Information",
            "warning.high_cpu": "High CPU usage detected",
            "warning.memory_low": "Low memory warning",
            "warning.connection_lost": "Connection lost",
            
            # User interface
            "ui.loading": "Loading...",
            "ui.please_wait": "Please wait",
            "ui.success": "Success",
            "ui.error": "Error",
            "ui.cancel": "Cancel",
            "ui.confirm": "Confirm",
            "ui.retry": "Retry",
            
            # Units and measurements
            "units.temperature": "°C",
            "units.pressure": "bar",
            "units.flow_rate": "L/min",
            "units.concentration": "ppm",
            "units.percentage": "%",
            "units.time_seconds": "seconds",
            "units.time_minutes": "minutes",
            "units.time_hours": "hours",
            
            # Compliance and regulations
            "compliance.gmp_compliant": "GMP Compliant",
            "compliance.fda_approved": "FDA Approved", 
            "compliance.audit_required": "Audit Required",
            "compliance.documentation_complete": "Documentation Complete",
            "compliance.signature_required": "Digital Signature Required"
        }
        
        # Save base translations
        with open(self.config.translations_path / "en_US.json", 'w', encoding='utf-8') as f:
            json.dump(base_translations, f, indent=2, ensure_ascii=False)
        
        # Create template translations for other locales
        locale_templates = {
            SupportedLocale.SPANISH_ES: {
                "system.startup": "Sistema iniciándose",
                "system.ready": "Sistema listo",
                "quality.batch_approved": "Lote aprobado para liberación",
                "quality.anomaly_detected": "Anomalía de calidad detectada",
                "sensor.connected": "Sensor conectado correctamente",
                "alert.critical": "Alerta Crítica",
                "ui.loading": "Cargando...",
                "units.temperature": "°C"
            },
            
            SupportedLocale.FRENCH_FR: {
                "system.startup": "Démarrage du système",
                "system.ready": "Système prêt",
                "quality.batch_approved": "Lot approuvé pour libération",
                "quality.anomaly_detected": "Anomalie qualité détectée",
                "sensor.connected": "Capteur connecté avec succès",
                "alert.critical": "Alerte Critique",
                "ui.loading": "Chargement...",
                "units.temperature": "°C"
            },
            
            SupportedLocale.GERMAN_DE: {
                "system.startup": "System startet",
                "system.ready": "System bereit",
                "quality.batch_approved": "Charge für Freigabe genehmigt",
                "quality.anomaly_detected": "Qualitätsanomalie erkannt",
                "sensor.connected": "Sensor erfolgreich verbunden",
                "alert.critical": "Kritische Warnung",
                "ui.loading": "Laden...",
                "units.temperature": "°C"
            },
            
            SupportedLocale.JAPANESE_JP: {
                "system.startup": "システムを開始しています",
                "system.ready": "システム準備完了",
                "quality.batch_approved": "バッチがリリース承認されました",
                "quality.anomaly_detected": "品質異常が検出されました",
                "sensor.connected": "センサーが正常に接続されました",
                "alert.critical": "重要なアラート",
                "ui.loading": "読み込み中...",
                "units.temperature": "°C"
            },
            
            SupportedLocale.CHINESE_CN: {
                "system.startup": "系统正在启动",
                "system.ready": "系统就绪",
                "quality.batch_approved": "批次已批准发布",
                "quality.anomaly_detected": "检测到质量异常",
                "sensor.connected": "传感器连接成功",
                "alert.critical": "严重警报",
                "ui.loading": "加载中...",
                "units.temperature": "°C"
            }
        }
        
        # Save template translations
        for locale, translations in locale_templates.items():
            with open(self.config.translations_path / f"{locale.value}.json", 'w', encoding='utf-8') as f:
                json.dump(translations, f, indent=2, ensure_ascii=False)
    
    def _auto_detect_locale(self):
        """Auto-detect system locale."""
        try:
            import locale as system_locale
            system_lang = system_locale.getdefaultlocale()[0]
            
            if system_lang:
                # Map system locale to supported locale
                locale_mapping = {
                    'en_US': SupportedLocale.ENGLISH_US,
                    'es_ES': SupportedLocale.SPANISH_ES,
                    'fr_FR': SupportedLocale.FRENCH_FR,
                    'de_DE': SupportedLocale.GERMAN_DE,
                    'ja_JP': SupportedLocale.JAPANESE_JP,
                    'zh_CN': SupportedLocale.CHINESE_CN,
                    'zh_TW': SupportedLocale.CHINESE_TW,
                    'ko_KR': SupportedLocale.KOREAN_KR,
                    'it_IT': SupportedLocale.ITALIAN_IT,
                    'pt_BR': SupportedLocale.PORTUGUESE_BR,
                    'ru_RU': SupportedLocale.RUSSIAN_RU,
                    'hi_IN': SupportedLocale.HINDI_IN
                }
                
                if system_lang in locale_mapping:
                    self.current_locale = locale_mapping[system_lang]
                    self.logger.info(f"Auto-detected locale: {self.current_locale.value}")
        except Exception as e:
            self.logger.debug(f"Failed to auto-detect locale: {e}")
    
    def set_locale(self, locale: SupportedLocale):
        """Set the current locale."""
        self.current_locale = locale
        self.logger.info(f"Locale set to: {locale.value}")
    
    def get_current_locale(self) -> SupportedLocale:
        """Get the current locale."""
        return self.current_locale
    
    def translate(self, key: str, **kwargs) -> str:
        """
        Translate a key to the current locale.
        
        Args:
            key: Translation key
            **kwargs: Variables for string formatting
            
        Returns:
            Translated string
        """
        # Try current locale first
        current_translations = self.translations.get(self.current_locale.value, {})
        if key in current_translations:
            translated = current_translations[key]
        else:
            # Fall back to default locale
            fallback_translations = self.translations.get(self.config.fallback_locale.value, {})
            translated = fallback_translations.get(key, key)  # Use key as fallback
            
            if key not in current_translations:
                self.logger.debug(f"Translation missing for key '{key}' in locale '{self.current_locale.value}'")
        
        # Apply string formatting if variables provided
        if kwargs:
            try:
                translated = translated.format(**kwargs)
            except Exception as e:
                self.logger.error(f"Error formatting translation '{key}': {e}")
        
        return translated
    
    def format_number(self, number: float, decimals: int = 2) -> str:
        """Format number according to current locale."""
        locale_formats = {
            SupportedLocale.ENGLISH_US: f"{{:,.{decimals}f}}",
            SupportedLocale.GERMAN_DE: f"{{:,.{decimals}f}}".replace(",", "X").replace(".", ",").replace("X", "."),
            SupportedLocale.FRENCH_FR: f"{{:,.{decimals}f}}".replace(",", " ").replace(".", ","),
            # Add more locale-specific number formats as needed
        }
        
        format_str = locale_formats.get(self.current_locale, f"{{:,.{decimals}f}}")
        
        try:
            return format_str.format(number)
        except:
            return str(number)
    
    def format_date(self, date, format_type: str = "short") -> str:
        """Format date according to current locale."""
        date_formats = {
            SupportedLocale.ENGLISH_US: {
                "short": "%m/%d/%Y",
                "medium": "%b %d, %Y",
                "long": "%B %d, %Y"
            },
            SupportedLocale.GERMAN_DE: {
                "short": "%d.%m.%Y",
                "medium": "%d. %b %Y",
                "long": "%d. %B %Y"
            },
            SupportedLocale.FRENCH_FR: {
                "short": "%d/%m/%Y",
                "medium": "%d %b %Y",
                "long": "%d %B %Y"
            },
            SupportedLocale.JAPANESE_JP: {
                "short": "%Y/%m/%d",
                "medium": "%Y年%m月%d日",
                "long": "%Y年%m月%d日"
            }
        }
        
        locale_formats = date_formats.get(self.current_locale, date_formats[SupportedLocale.ENGLISH_US])
        format_str = locale_formats.get(format_type, locale_formats["short"])
        
        try:
            return date.strftime(format_str)
        except:
            return str(date)
    
    def get_available_locales(self) -> List[SupportedLocale]:
        """Get list of available locales with translations."""
        available = []
        for locale in SupportedLocale:
            if locale.value in self.translations and self.translations[locale.value]:
                available.append(locale)
        return available
    
    def get_translation_coverage(self, locale: SupportedLocale) -> float:
        """Get translation coverage percentage for a locale."""
        if self.config.fallback_locale.value not in self.translations:
            return 0.0
        
        base_keys = set(self.translations[self.config.fallback_locale.value].keys())
        locale_keys = set(self.translations.get(locale.value, {}).keys())
        
        if not base_keys:
            return 100.0
        
        coverage = len(locale_keys & base_keys) / len(base_keys)
        return coverage * 100.0
    
    def add_translation(self, locale: SupportedLocale, key: str, value: str):
        """Add a translation for a specific locale."""
        if locale.value not in self.translations:
            self.translations[locale.value] = {}
        
        self.translations[locale.value][key] = value
        self.logger.debug(f"Added translation for {locale.value}: {key}")
    
    def export_translations(self, locale: SupportedLocale) -> Dict[str, str]:
        """Export translations for a locale."""
        return self.translations.get(locale.value, {}).copy()
    
    def import_translations(self, locale: SupportedLocale, translations: Dict[str, str]):
        """Import translations for a locale."""
        if locale.value not in self.translations:
            self.translations[locale.value] = {}
        
        self.translations[locale.value].update(translations)
        self.logger.info(f"Imported {len(translations)} translations for {locale.value}")


# Global i18n instance
_i18n_manager = None


def get_i18n_manager() -> I18nManager:
    """Get global i18n manager instance."""
    global _i18n_manager
    if _i18n_manager is None:
        _i18n_manager = I18nManager()
    return _i18n_manager


def t(key: str, **kwargs) -> str:
    """Convenience function for translation."""
    return get_i18n_manager().translate(key, **kwargs)


def set_global_locale(locale: SupportedLocale):
    """Set global locale."""
    get_i18n_manager().set_locale(locale)


def format_number_locale(number: float, decimals: int = 2) -> str:
    """Format number according to current locale."""
    return get_i18n_manager().format_number(number, decimals)


def format_date_locale(date, format_type: str = "short") -> str:
    """Format date according to current locale."""
    return get_i18n_manager().format_date(date, format_type)