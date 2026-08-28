import sys
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from typing import Any, Optional


class CliExitCodes(IntEnum):
    SUCCESS = 0
    
    INVALID_ARGS = 1
    MISSING_ARGUMENT = 2
    INVALID_OPTION = 3
    INVALID_VALUE = 4
    MISSING_REQUIRED_FLAG = 5
    INVALID_ARGUMENT_FORMAT = 6
    UNKNOWN_COMMAND = 7
    INVALID_COMMAND = 8
    DUPLICATE_ARGUMENT = 9
    INVALID_ARGUMENT_COMBINATION = 10
    ARGUMENT_TOO_LONG = 11
    ARGUMENT_TOO_SHORT = 12
    INVALID_ENUM_VALUE = 13
    PARSING_ERROR = 14
    INVALID_INTERVAL = 15
    INVALID_RANGE = 16
    INVALID_REGEX = 17
    INVALID_JSON = 18
    INVALID_YAML = 19
    
    INVALID_FOLDER = 20
    FILE_NOT_FOUND = 21
    FOLDER_NOT_FOUND = 22
    INVALID_FILE_PATH = 23
    FILE_EXISTS = 24
    FOLDER_EXISTS = 25
    FILE_EMPTY = 26
    FILE_TOO_LARGE = 27
    FILE_NOT_READABLE = 28
    FILE_NOT_WRITABLE = 29
    FILE_CORRUPTED = 30
    FILE_FORMAT_ERROR = 31
    FILE_PERMISSION_ERROR = 32
    FOLDER_PERMISSION_ERROR = 33
    PATH_TOO_LONG = 34
    SYMLINK_ERROR = 35
    FILE_LOCKED = 36
    FILE_IN_USE = 37
    FILE_ENCODING_ERROR = 38
    UNSUPPORTED_FILE_TYPE = 39
    
    FATAL_ERROR = 40
    RUNTIME_ERROR = 41
    PROCESS_ERROR = 42
    SUBPROCESS_ERROR = 43
    TIMEOUT_ERROR = 44
    MEMORY_ERROR = 45
    CPU_LIMIT_ERROR = 46
    DISK_SPACE_ERROR = 47
    IO_ERROR = 48
    UNHANDLED_EXCEPTION = 49
    OPERATION_CANCELLED = 50
    OPERATION_ABORTED = 51
    WORKER_ERROR = 52
    QUEUE_ERROR = 53
    THREAD_ERROR = 54
    LOCK_ERROR = 55
    SYNC_ERROR = 56
    DAEMON_ERROR = 57
    SERVICE_ERROR = 58
    INTERRUPTED = 59
    
    NETWORK_ERROR = 60
    CONNECTION_ERROR = 61
    CONNECTION_TIMEOUT = 62
    CONNECTION_REFUSED = 63
    CONNECTION_RESET = 64
    HOST_UNREACHABLE = 65
    PORT_UNAVAILABLE = 66
    DNS_ERROR = 67
    SSL_ERROR = 68
    AUTHENTICATION_ERROR = 69
    AUTHORIZATION_ERROR = 70
    RATE_LIMIT_ERROR = 71
    API_ERROR = 72
    API_TIMEOUT = 73
    WEBHOOK_ERROR = 74
    PROXY_ERROR = 75
    TUNNEL_ERROR = 76
    FIREWALL_ERROR = 77
    LOAD_BALANCER_ERROR = 78
    GATEWAY_ERROR = 79
    
    SYSTEM_ERROR = 80
    ENVIRONMENT_ERROR = 81
    PYTHON_VERSION_ERROR = 82
    DEPENDENCY_MISSING = 83
    DEPENDENCY_VERSION = 84
    LIBRARY_ERROR = 85
    DRIVER_ERROR = 86
    KERNEL_ERROR = 87
    OS_ERROR = 88
    PLATFORM_ERROR = 89
    PACKAGE_ERROR = 90
    MODULE_ERROR = 91
    CLASS_ERROR = 92
    INITIALIZATION_ERROR = 93
    SHUTDOWN_ERROR = 94
    CLEANUP_ERROR = 95
    RESOURCE_ERROR = 96
    RESOURCE_LIMIT = 97
    ENV_VAR_MISSING = 98
    ENV_VAR_INVALID = 99
    
    CONFIG_ERROR = 100
    CONFIG_FILE_MISSING = 101
    CONFIG_FILE_INVALID = 102
    CONFIG_KEY_MISSING = 103
    CONFIG_VALUE_INVALID = 104
    CONFIG_SYNTAX_ERROR = 105
    CONFIG_PERMISSION_ERROR = 106
    CONFIG_LOAD_ERROR = 107
    CONFIG_SAVE_ERROR = 108
    CONFIG_MIGRATION_ERROR = 109
    CONFIG_BACKUP_ERROR = 110
    CONFIG_RESTORE_ERROR = 111
    CONFIG_INCOMPATIBLE = 112
    CONFIG_CORRUPTED = 113
    CONFIG_AMBIGUOUS = 114
    CONFIG_DEPRECATED = 115
    CONFIG_NOT_FOUND = 116
    CONFIG_DUPLICATE = 117
    CONFIG_ENCRYPTION_ERROR = 118
    CONFIG_DECRYPTION_ERROR = 119
    
    DATA_ERROR = 120
    DATA_CORRUPTED = 121
    DATA_MISSING = 122
    DATA_INVALID = 123
    DATA_INCOMPLETE = 124
    DATA_OUTDATED = 125
    DATA_INCONSISTENT = 126
    DATA_DUPLICATE = 127
    DATA_OVERFLOW = 128
    DATA_UNDERFLOW = 129
    DATA_CONVERSION_ERROR = 130
    DATA_ENCODING_ERROR = 131
    DATA_DECODING_ERROR = 132
    DATA_SERIALIZATION_ERROR = 133
    DATA_DESERIALIZATION_ERROR = 134
    DATA_VALIDATION_ERROR = 135
    DATA_TRANSFORMATION_ERROR = 136
    DATA_AGGREGATION_ERROR = 137
    DATA_NORMALIZATION_ERROR = 138
    DATA_PREPROCESSING_ERROR = 139
    
    PERMISSION_ERROR = 140
    PERMISSION_DENIED = 141
    ACCESS_DENIED = 142
    ACCESS_FORBIDDEN = 143
    UNAUTHORIZED = 144
    SECURITY_ERROR = 145
    ENCRYPTION_ERROR = 146
    DECRYPTION_ERROR = 147
    HASH_ERROR = 148
    SIGNATURE_ERROR = 149
    CERTIFICATE_ERROR = 150
    TOKEN_ERROR = 151
    TOKEN_EXPIRED = 152
    TOKEN_INVALID = 153
    SESSION_ERROR = 154
    SESSION_EXPIRED = 155
    SESSION_INVALID = 156
    CSRF_ERROR = 157
    CORS_ERROR = 158
    SECURITY_POLICY_ERROR = 159
    
    EXTERNAL_SERVICE_ERROR = 160
    DATABASE_ERROR = 161
    DATABASE_CONNECTION_ERROR = 162
    DATABASE_QUERY_ERROR = 163
    DATABASE_TIMEOUT = 164
    DATABASE_LOCK_ERROR = 165
    CACHE_ERROR = 166
    CACHE_MISS = 167
    CACHE_EXPIRED = 168
    MESSAGE_QUEUE_ERROR = 169
    FILE_SYSTEM_ERROR = 170
    STORAGE_ERROR = 171
    STORAGE_QUOTA_EXCEEDED = 172
    BACKUP_ERROR = 173
    RESTORE_ERROR = 174
    REPLICATION_ERROR = 175
    SYNC_ERROR_EXTERNAL = 176
    IMPORT_ERROR = 177
    EXPORT_ERROR = 178
    MIGRATION_ERROR = 179
    
    INVALID_FOLDER_ALIAS = 1
    FILE_NOT_FOUND_ALIAS = 40
    INVALID_COMMAND_ALIAS = 1
    
    @classmethod
    def get_description(cls, code: int) -> str:
        descriptions = {
            cls.SUCCESS: "Success",
            cls.INVALID_ARGS: "Invalid arguments",
            cls.MISSING_ARGUMENT: "Missing required argument",
            cls.INVALID_OPTION: "Invalid option",
            cls.INVALID_VALUE: "Invalid value",
            cls.MISSING_REQUIRED_FLAG: "Missing required flag",
            cls.INVALID_ARGUMENT_FORMAT: "Invalid argument format",
            cls.UNKNOWN_COMMAND: "Unknown command",
            cls.INVALID_COMMAND: "Invalid command",
            cls.DUPLICATE_ARGUMENT: "Duplicate argument",
            cls.INVALID_ARGUMENT_COMBINATION: "Invalid argument combination",
            cls.ARGUMENT_TOO_LONG: "Argument too long",
            cls.ARGUMENT_TOO_SHORT: "Argument too short",
            cls.INVALID_ENUM_VALUE: "Invalid enum value",
            cls.PARSING_ERROR: "Parsing error",
            cls.INVALID_INTERVAL: "Invalid interval",
            cls.INVALID_RANGE: "Invalid range",
            cls.INVALID_REGEX: "Invalid regular expression",
            cls.INVALID_JSON: "Invalid JSON",
            cls.INVALID_YAML: "Invalid YAML",
            cls.INVALID_FOLDER: "Invalid folder",
            cls.FILE_NOT_FOUND: "File not found",
            cls.FOLDER_NOT_FOUND: "Folder not found",
            cls.INVALID_FILE_PATH: "Invalid file path",
            cls.FILE_EXISTS: "File already exists",
            cls.FOLDER_EXISTS: "Folder already exists",
            cls.FILE_EMPTY: "File is empty",
            cls.FILE_TOO_LARGE: "File too large",
            cls.FILE_NOT_READABLE: "File not readable",
            cls.FILE_NOT_WRITABLE: "File not writable",
            cls.FILE_CORRUPTED: "File corrupted",
            cls.FILE_FORMAT_ERROR: "File format error",
            cls.FILE_PERMISSION_ERROR: "File permission error",
            cls.FOLDER_PERMISSION_ERROR: "Folder permission error",
            cls.PATH_TOO_LONG: "Path too long",
            cls.SYMLINK_ERROR: "Symbolic link error",
            cls.FILE_LOCKED: "File locked",
            cls.FILE_IN_USE: "File in use",
            cls.FILE_ENCODING_ERROR: "File encoding error",
            cls.UNSUPPORTED_FILE_TYPE: "Unsupported file type",
            cls.FATAL_ERROR: "Fatal error",
            cls.RUNTIME_ERROR: "Runtime error",
            cls.PROCESS_ERROR: "Process error",
            cls.SUBPROCESS_ERROR: "Subprocess error",
            cls.TIMEOUT_ERROR: "Timeout error",
            cls.MEMORY_ERROR: "Memory error",
            cls.CPU_LIMIT_ERROR: "CPU limit exceeded",
            cls.DISK_SPACE_ERROR: "Insufficient disk space",
            cls.IO_ERROR: "Input/Output error",
            cls.UNHANDLED_EXCEPTION: "Unhandled exception",
            cls.OPERATION_CANCELLED: "Operation cancelled",
            cls.OPERATION_ABORTED: "Operation aborted",
            cls.WORKER_ERROR: "Worker error",
            cls.QUEUE_ERROR: "Queue error",
            cls.THREAD_ERROR: "Thread error",
            cls.LOCK_ERROR: "Lock error",
            cls.SYNC_ERROR: "Synchronization error",
            cls.DAEMON_ERROR: "Daemon error",
            cls.SERVICE_ERROR: "Service error",
            cls.INTERRUPTED: "Interrupted",
            cls.NETWORK_ERROR: "Network error",
            cls.CONNECTION_ERROR: "Connection error",
            cls.CONNECTION_TIMEOUT: "Connection timeout",
            cls.CONNECTION_REFUSED: "Connection refused",
            cls.CONNECTION_RESET: "Connection reset",
            cls.HOST_UNREACHABLE: "Host unreachable",
            cls.PORT_UNAVAILABLE: "Port unavailable",
            cls.DNS_ERROR: "DNS error",
            cls.SSL_ERROR: "SSL/TLS error",
            cls.AUTHENTICATION_ERROR: "Authentication error",
            cls.AUTHORIZATION_ERROR: "Authorization error",
            cls.RATE_LIMIT_ERROR: "Rate limit exceeded",
            cls.API_ERROR: "API error",
            cls.API_TIMEOUT: "API timeout",
            cls.WEBHOOK_ERROR: "Webhook error",
            cls.PROXY_ERROR: "Proxy error",
            cls.TUNNEL_ERROR: "Tunnel error",
            cls.FIREWALL_ERROR: "Firewall error",
            cls.LOAD_BALANCER_ERROR: "Load balancer error",
            cls.GATEWAY_ERROR: "Gateway error",
            cls.SYSTEM_ERROR: "System error",
            cls.ENVIRONMENT_ERROR: "Environment error",
            cls.PYTHON_VERSION_ERROR: "Python version error",
            cls.DEPENDENCY_MISSING: "Missing dependency",
            cls.DEPENDENCY_VERSION: "Dependency version error",
            cls.LIBRARY_ERROR: "Library error",
            cls.DRIVER_ERROR: "Driver error",
            cls.KERNEL_ERROR: "Kernel error",
            cls.OS_ERROR: "Operating system error",
            cls.PLATFORM_ERROR: "Platform error",
            cls.PACKAGE_ERROR: "Package error",
            cls.MODULE_ERROR: "Module error",
            cls.CLASS_ERROR: "Class error",
            cls.INITIALIZATION_ERROR: "Initialization error",
            cls.SHUTDOWN_ERROR: "Shutdown error",
            cls.CLEANUP_ERROR: "Cleanup error",
            cls.RESOURCE_ERROR: "Resource error",
            cls.RESOURCE_LIMIT: "Resource limit exceeded",
            cls.ENV_VAR_MISSING: "Environment variable missing",
            cls.ENV_VAR_INVALID: "Environment variable invalid",
            cls.CONFIG_ERROR: "Configuration error",
            cls.CONFIG_FILE_MISSING: "Config file not found",
            cls.CONFIG_FILE_INVALID: "Config file invalid",
            cls.CONFIG_KEY_MISSING: "Config key missing",
            cls.CONFIG_VALUE_INVALID: "Config value invalid",
            cls.CONFIG_SYNTAX_ERROR: "Config syntax error",
            cls.CONFIG_PERMISSION_ERROR: "Config permission error",
            cls.CONFIG_LOAD_ERROR: "Config load error",
            cls.CONFIG_SAVE_ERROR: "Config save error",
            cls.CONFIG_MIGRATION_ERROR: "Config migration error",
            cls.CONFIG_BACKUP_ERROR: "Config backup error",
            cls.CONFIG_RESTORE_ERROR: "Config restore error",
            cls.CONFIG_INCOMPATIBLE: "Incompatible config",
            cls.CONFIG_CORRUPTED: "Config corrupted",
            cls.CONFIG_AMBIGUOUS: "Ambiguous config",
            cls.CONFIG_DEPRECATED: "Deprecated config",
            cls.CONFIG_NOT_FOUND: "Config not found",
            cls.CONFIG_DUPLICATE: "Duplicate config",
            cls.CONFIG_ENCRYPTION_ERROR: "Config encryption error",
            cls.CONFIG_DECRYPTION_ERROR: "Config decryption error",
            cls.DATA_ERROR: "Data error",
            cls.DATA_CORRUPTED: "Data corrupted",
            cls.DATA_MISSING: "Data missing",
            cls.DATA_INVALID: "Data invalid",
            cls.DATA_INCOMPLETE: "Data incomplete",
            cls.DATA_OUTDATED: "Data outdated",
            cls.DATA_INCONSISTENT: "Data inconsistent",
            cls.DATA_DUPLICATE: "Data duplicate",
            cls.DATA_OVERFLOW: "Data overflow",
            cls.DATA_UNDERFLOW: "Data underflow",
            cls.DATA_CONVERSION_ERROR: "Data conversion error",
            cls.DATA_ENCODING_ERROR: "Data encoding error",
            cls.DATA_DECODING_ERROR: "Data decoding error",
            cls.DATA_SERIALIZATION_ERROR: "Data serialization error",
            cls.DATA_DESERIALIZATION_ERROR: "Data deserialization error",
            cls.DATA_VALIDATION_ERROR: "Data validation error",
            cls.DATA_TRANSFORMATION_ERROR: "Data transformation error",
            cls.DATA_AGGREGATION_ERROR: "Data aggregation error",
            cls.DATA_NORMALIZATION_ERROR: "Data normalization error",
            cls.DATA_PREPROCESSING_ERROR: "Data preprocessing error",
            cls.PERMISSION_ERROR: "Permission error",
            cls.PERMISSION_DENIED: "Permission denied",
            cls.ACCESS_DENIED: "Access denied",
            cls.ACCESS_FORBIDDEN: "Access forbidden",
            cls.UNAUTHORIZED: "Unauthorized",
            cls.SECURITY_ERROR: "Security error",
            cls.ENCRYPTION_ERROR: "Encryption error",
            cls.DECRYPTION_ERROR: "Decryption error",
            cls.HASH_ERROR: "Hash error",
            cls.SIGNATURE_ERROR: "Signature error",
            cls.CERTIFICATE_ERROR: "Certificate error",
            cls.TOKEN_ERROR: "Token error",
            cls.TOKEN_EXPIRED: "Token expired",
            cls.TOKEN_INVALID: "Token invalid",
            cls.SESSION_ERROR: "Session error",
            cls.SESSION_EXPIRED: "Session expired",
            cls.SESSION_INVALID: "Session invalid",
            cls.CSRF_ERROR: "CSRF error",
            cls.CORS_ERROR: "CORS error",
            cls.SECURITY_POLICY_ERROR: "Security policy error",
            cls.EXTERNAL_SERVICE_ERROR: "External service error",
            cls.DATABASE_ERROR: "Database error",
            cls.DATABASE_CONNECTION_ERROR: "Database connection error",
            cls.DATABASE_QUERY_ERROR: "Database query error",
            cls.DATABASE_TIMEOUT: "Database timeout",
            cls.DATABASE_LOCK_ERROR: "Database lock error",
            cls.CACHE_ERROR: "Cache error",
            cls.CACHE_MISS: "Cache miss",
            cls.CACHE_EXPIRED: "Cache expired",
            cls.MESSAGE_QUEUE_ERROR: "Message queue error",
            cls.FILE_SYSTEM_ERROR: "File system error",
            cls.STORAGE_ERROR: "Storage error",
            cls.STORAGE_QUOTA_EXCEEDED: "Storage quota exceeded",
            cls.BACKUP_ERROR: "Backup error",
            cls.RESTORE_ERROR: "Restore error",
            cls.REPLICATION_ERROR: "Replication error",
            cls.SYNC_ERROR_EXTERNAL: "Synchronization error",
            cls.IMPORT_ERROR: "Import error",
            cls.EXPORT_ERROR: "Export error",
            cls.MIGRATION_ERROR: "Migration error",
        }
        return descriptions.get(cls(code), "Unknown error")
    
    @classmethod
    def get_exit_code(cls, code: int) -> Optional['CliExitCodes']:
        try:
            return cls(code)
        except ValueError:
            return None
    
    @classmethod
    def is_success(cls, code: int) -> bool:
        return code == cls.SUCCESS
    
    @classmethod
    def is_error(cls, code: int) -> bool:
        return code != cls.SUCCESS
    
    @classmethod
    def is_invalid_args(cls, code: int) -> bool:
        return code in range(1, 20)
    
    @classmethod
    def is_file_error(cls, code: int) -> bool:
        return code in range(20, 40)
    
    @classmethod
    def is_runtime_error(cls, code: int) -> bool:
        return code in range(40, 60)
    
    @classmethod
    def is_network_error(cls, code: int) -> bool:
        return code in range(60, 80)
    
    @classmethod
    def is_system_error(cls, code: int) -> bool:
        return code in range(80, 100)
    
    @classmethod
    def is_config_error(cls, code: int) -> bool:
        return code in range(100, 120)
    
    @classmethod
    def is_data_error(cls, code: int) -> bool:
        return code in range(120, 140)
    
    @classmethod
    def is_permission_error(cls, code: int) -> bool:
        return code in range(140, 160)
    
    @classmethod
    def is_external_error(cls, code: int) -> bool:
        return code in range(160, 180)
    
    @classmethod
    def get_category(cls, code: int) -> str:
        categories = {
            cls.SUCCESS: "Success",
            **{i: "Argument/Input Error" for i in range(1, 20)},
            **{i: "File/Path Error" for i in range(20, 40)},
            **{i: "Runtime Error" for i in range(40, 60)},
            **{i: "Network Error" for i in range(60, 80)},
            **{i: "System/Environment Error" for i in range(80, 100)},
            **{i: "Configuration Error" for i in range(100, 120)},
            **{i: "Data/Processing Error" for i in range(120, 140)},
            **{i: "Permission/Security Error" for i in range(140, 160)},
            **{i: "External Service Error" for i in range(160, 180)},
        }
        return categories.get(code, "Unknown Category")
    
    @classmethod
    def get_exit_code_name(cls, code: int) -> str:
        try:
            return cls(code).name
        except ValueError:
            return "UNKNOWN"
    
    @classmethod
    def to_dict(cls) -> dict[int, str]:
        return {code.value: code.name for code in cls}
    
    @classmethod
    def to_json(cls) -> str:
        import json
        return json.dumps(cls.to_dict(), indent=2)
    
    @classmethod
    def print_all(cls) -> None:
        print(f"{'Code':<10} {'Name':<40} {'Description':<60}")
        print("-" * 110)
        for code in cls:
            desc = cls.get_description(code.value)
            print(f"{code.value:<10} {code.name:<40} {desc:<60}")


@dataclass
class ExitInfo:
    code: int
    message: str
    category: str
    timestamp: str
    
    @classmethod
    def from_code(cls, code: int, message: str = "") -> 'ExitInfo':
        return cls(
            code=code,
            message=message or CliExitCodes.get_description(code),
            category=CliExitCodes.get_category(code),
            timestamp=datetime.now().isoformat()
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "category": self.category,
            "timestamp": self.timestamp
        }


def exit_with_code(code: int, message: Optional[str] = None) -> None:
    if message:
        print(f"Error: {message}", file=sys.stderr)
    else:
        print(f"Error: {CliExitCodes.get_description(code)}", file=sys.stderr)
    sys.exit(code)


def get_exit_code(name: str) -> Optional[int]:
    try:
        return CliExitCodes[name].value
    except KeyError:
        return None


def list_all_exit_codes() -> None:
    print("Available exit codes:")
    for code in CliExitCodes:
        print(f"  {code.value:>3} - {code.name}")


class ExitCodeManager:
    def __init__(self):
        self._exit_codes = CliExitCodes
    
    def get_code(self, name: str) -> Optional[int]:
        return get_exit_code(name)
    
    def get_name(self, code: int) -> str:
        return CliExitCodes.get_exit_code_name(code)
    
    def get_description(self, code: int) -> str:
        return CliExitCodes.get_description(code)
    
    def get_category(self, code: int) -> str:
        return CliExitCodes.get_category(code)
    
    def is_success(self, code: int) -> bool:
        return CliExitCodes.is_success(code)
    
    def is_error(self, code: int) -> bool:
        return CliExitCodes.is_error(code)
    
    def is_invalid_args(self, code: int) -> bool:
        return CliExitCodes.is_invalid_args(code)
    
    def is_file_error(self, code: int) -> bool:
        return CliExitCodes.is_file_error(code)
    
    def is_runtime_error(self, code: int) -> bool:
        return CliExitCodes.is_runtime_error(code)
    
    def is_network_error(self, code: int) -> bool:
        return CliExitCodes.is_network_error(code)
    
    def is_system_error(self, code: int) -> bool:
        return CliExitCodes.is_system_error(code)
    
    def is_config_error(self, code: int) -> bool:
        return CliExitCodes.is_config_error(code)
    
    def is_data_error(self, code: int) -> bool:
        return CliExitCodes.is_data_error(code)
    
    def is_permission_error(self, code: int) -> bool:
        return CliExitCodes.is_permission_error(code)
    
    def is_external_error(self, code: int) -> bool:
        return CliExitCodes.is_external_error(code)
    
    def print_all(self) -> None:
        CliExitCodes.print_all()
    
    def to_dict(self) -> dict[int, str]:
        return CliExitCodes.to_dict()
    
    def to_json(self) -> str:
        return CliExitCodes.to_json()


class ExitHandler:
    @staticmethod
    def handle(code: int, message: Optional[str] = None) -> None:
        exit_with_code(code, message)
    
    @staticmethod
    def handle_with_info(code: int, message: Optional[str] = None) -> None:
        info = ExitInfo.from_code(code, message or "")
        print(f"Exit: {info.to_dict()}", file=sys.stderr)
        sys.exit(code)
    
    @staticmethod
    def success(message: str = "Success") -> None:
        print(message)
        sys.exit(CliExitCodes.SUCCESS)
    
    @staticmethod
    def error(code: int, message: str) -> None:
        print(f"Error [{code}]: {message}", file=sys.stderr)
        sys.exit(code)


__all__ = [
    "CliExitCodes",
    "ExitInfo",
    "exit_with_code",
    "get_exit_code",
    "list_all_exit_codes",
    "ExitCodeManager",
    "ExitHandler"
]
